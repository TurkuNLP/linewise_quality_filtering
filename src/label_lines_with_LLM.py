# Standard libraries
import argparse
from collections import Counter
import csv
import datetime
import json
import logging
import math
import numpy as np
import os
from pathlib import Path
from random import shuffle
import re
import time

# Third party imports
from datasets import load_dataset  # type: ignore
import json_repair # type: ignore
from pydantic import RootModel  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from sklearn.cluster import AgglomerativeClustering  # type: ignore
import torch  # type: ignore
import torch.distributed as dist  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

# Local imports
from embed import StellaEmbedder
import prompts

# Configure logging
slurm_job_id = os.environ.get("SLURM_JOB_ID", "default_id")
logging.basicConfig(
    filename=f"../logs/{slurm_job_id}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


class LineClassifier:

    def __init__(self, args):
        self.cache_dir = args.cache_dir or os.environ["HF_HOME"]
        self.run_id = args.run_id
        self.model_name = args.model_name
        self.temperature = args.temperature
        self.max_vocab = args.max_vocab
        self.synonym_threshold = args.synonym_threshold
        self.start_index = args.start_index
        self.stop_index = args.stop_index
        self.batch_size = args.batch_size
        self.results_dir = Path(args.results_dir)
        self.language = args.language
        self.use_fixed_labels = args.use_fixed_labels
        self.label_counts = self.load_previous_labels()
        self.embedder = StellaEmbedder(self.cache_dir)

    def model_setup(self):
        self.model = LLM(
            model=self.model_name,
            download_dir=self.cache_dir,
            dtype="bfloat16",
            max_model_len=128_000,
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=False,
            gpu_memory_utilization=0.8,
        )

    def classification_labels(self):
        return [
            "Clean",
            "Formatting, Style & Errors",
            "Bibliographical & Citation References",
            "Promotional & Spam Content",
            "Contact & Identification Information",
            "Navigation & Interface Elements",
            "Technical Specifications & Metadata",
            "Legal & Administrative Content",
            "Offensive or Inappropriate Content",
        ]

    def generate(self, model_input, response_schema):
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.5,
            repetition_penalty=1,  # 1 = no penalty, >1 penalty
            max_tokens=3000,  # max tokens to generate
            guided_decoding=response_schema,
        )

        batched_outputs = self.model.generate(
            model_input, sampling_params=sampling_params, use_tqdm=False
        )
        
        outputs = [out.outputs[0].text.strip(" `\njson") for out in batched_outputs]

        validated_outputs = [json_repair.loads(output) for output in outputs]
        
        return validated_outputs

    def format_input(
        self, input_lines=None, vocab=None, group_name=None, synonyms=None, task=None
    ):
        """Format input lines."""
        
        # Get language from the language code
        language = self.code_to_language()

        if task == "generate_labels" or task == "classify":
            formatted_lines = ""
            for i, line in enumerate(input_lines):
                formatted_lines += f"Line {i+1}: {line}\n------\n"

            if task == "classify":
                prompt = prompts.classify(formatted_lines, language)
                return prompt
            
            if not vocab:
                vocab = (
                    "The list is currently empty. You are free to create new labels."
                )
            prompt = prompts.generate_labels(formatted_lines, vocab, language)
            return prompt
        
        elif task == "synonyms":
            prompt = prompts.review_synonyms(group_name, synonyms)
            return prompt

    def batched(self, doc):
        lines = doc.strip().split("\n")
        return np.array_split(lines, math.ceil(len(lines) / self.batch_size))

    def extract_junk_labels(self, batches: list[dict]) -> list:
        junk_labels = []
        
        for batch in batches:
            for label in batch.values():
                if label.lower().strip() != "clean":
                    junk_labels.append(label)

        return junk_labels

    def split_long_line_into_segments(self, text):
        """
        Some documents contain only one, often long, line, which can cause issues.
        This function splits long lines into smaller segments.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)

        segments = []
        current_segment = ""

        for sentence in sentences:
            # If adding the sentence would exceed 200 chars, start a new segment
            if len(current_segment) + len(sentence) + 1 > 200:
                segments.append(
                    current_segment.strip()
                )  # Strip to remove any trailing spaces
                current_segment = sentence
            else:
                current_segment += " " + sentence

        # Add the last segment if it's non-empty
        if current_segment:
            segments.append(current_segment.strip())

        return segments

    def get_json_schema(self, task):
        schema_map = {
        "classify": dict[str, str],
        "generate_labels": dict[str, str],
        "synonyms": dict[str, list[str]],
        }
        return GuidedDecodingParams(json=RootModel[schema_map[task]].model_json_schema())

    def load_data(self):        
        return load_dataset("HPLT/HPLT2.0_cleaned",
                            name=self.language,
                            split="train",
                            streaming=True,
                            cache_dir=self.cache_dir
                            )
                    
    def load_previous_labels(self):
        junk_labels = Counter()
        file_path = self.results_dir / f"label_vocab_{self.run_id}.tsv"
        
        if file_path.exists():
            with open(file_path, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                junk_labels.update({row[0]: int(row[1]) for row in reader})
                
        return junk_labels

    def cluster_similar_labels(self, labels, embeddings):
        # Identify and remove zero vectors
        valid_indices = [
            i for i, vec in enumerate(embeddings) if np.linalg.norm(vec) > 0
        ]

        labels = [labels[i] for i in valid_indices]
        embeddings = embeddings[valid_indices]

        # Perform Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.synonym_threshold,
            metric="cosine",
            linkage="average",
        )
        cluster_assignments = clustering.fit_predict(embeddings)

        # Group words by cluster labels
        groups = {}
        for idx, label in enumerate(cluster_assignments):
            groups.setdefault(label, []).append(idx)

        # Find the medoid for each group
        group_dict = {}
        for label, indices in groups.items():
            group_vectors = embeddings[indices]
            distance_matrix = cdist(group_vectors, group_vectors, metric="cosine")
            medoid_index = np.argmin(np.sum(distance_matrix, axis=1))
            medoid_word = labels[indices[medoid_index]]

            # Store the group with the medoid as the key
            # Remove duplicates in each group
            group_dict[medoid_word] = list(set([labels[idx] for idx in indices]))

        return group_dict

    def evaluate_synonym_candidates(self, synonym_candidates):
        
        # Separate single-label and multi-label groups
        # Single-label groups can skip LLM evaluation
        single_label_groups = {k: v for k, v in synonym_candidates.items() if len(v) == 1}
        multi_label_groups = {k: v for k, v in synonym_candidates.items() if len(v) > 1}

        # Use LLM only for multi-label groups
        if multi_label_groups:
            prompts = [
                self.format_input(task="synonyms", group_name=group_name, synonyms=syns)
                for group_name, syns in multi_label_groups.items()
            ]

            json_schema = self.get_json_schema(task="synonyms")
            generated_groups = self.generate(prompts, json_schema)
        else:
            generated_groups = []

        # Make dictionary from LLM outputs
        generated_synonyms = {}
        for d in generated_groups:
            for key, value in d.items():
                if key in generated_synonyms:
                    generated_synonyms[key].extend(value)
                else:
                    generated_synonyms[key] = value
                    
        # Merge single-label groups and generated groups
        final_synonym_groups = {}

        for key, value in {**generated_synonyms, **single_label_groups}.items():
            if key in final_synonym_groups:
                final_synonym_groups[key] = list(set(final_synonym_groups[key] + value))  # Merge without duplicates
            else:
                final_synonym_groups[key] = value
                
        return final_synonym_groups

    def combine_synonyms(self, model_output: list[dict]):
        # Get a list of non-clean labels
        junk_labels = self.extract_junk_labels(model_output)
        
        # If there are not junk labels, we can skip synonym finding
        if len(junk_labels) == 0:
            return model_output

        # Append previous non-clean labels, if they exist
        file_path = self.results_dir / f"label_vocab_{self.run_id}.tsv"
        if file_path.exists():
            with open(file_path, "r") as f:
                file = f.readlines()
                junk_labels = [line.split("\t")[0] for line in file] + junk_labels

        # Remove duplicates
        junk_labels = list(set(junk_labels))
        
        # If there are still fewer than 2 junk labels, we can skip synonym finding.
        if len(junk_labels) < 2:
            return model_output
                
        # Embed labels
        embedder = StellaEmbedder()
        embeddings = embedder.embed_labels(junk_labels)

        # Group similar labels
        synonym_candidates = self.cluster_similar_labels(junk_labels, embeddings)
        
        synonyms = self.evaluate_synonym_candidates(synonym_candidates)

        replaced_output = self.replace_synonyms(synonyms, model_output)
        
        # Update previously saved results with new synonyms
        self.update_previous_results(synonyms)

        return replaced_output

    def replace_synonyms(self, synonyms, model_output):
        # Create a mapping from synonym to its group for fast lookup
        synonym_map = {
            syn: group for group, members in synonyms.items() for syn in members
        }

        # Replace synonyms
        for batch in model_output:
            for key, label in batch.items():
                if key == "line":
                    continue
                batch[key] = synonym_map.get(label, label)
            
        return model_output
    
    def update_previous_results(self, synonyms):
        file_path = self.results_dir / f"results_{self.run_id}.jsonl"
        
        if file_path.exists():
            with open(file_path, "r") as f:
                prev_results = [json.loads(line.strip()) for line in f.readlines()]
                
            for document in prev_results:
                document["content"] = self.replace_synonyms(synonyms, document["content"])
            
            with open(file_path, "w") as f:
                for doc in prev_results:
                    json_line = json.dumps(doc, ensure_ascii=False)
                    f.write(json_line + "\n")

    def format_results(self, label_lists, batches, valid_labels=None):
        results = []
    
        for line_nums_and_labels, batch in zip(label_lists, batches):
            for line_num, text_line in zip(line_nums_and_labels, batch):
                label = line_nums_and_labels.get(line_num, "Clean")
                # In the classification pipeline:
                # If model has generated an invalid label
                # we simply default to "Clean".
                # In the label generation pipeline:
                # The condition is never met and the if-block
                # never executed.
                if valid_labels and label not in valid_labels:
                    label = "Clean"
                dict = {"line": text_line,
                        "label": label}
                results.append(dict)
                
        return results
             
    def save_results(self, document, results):
        
        def serialize_datetime(obj): 
            if isinstance(obj, datetime.datetime): 
                return obj.isoformat() 
            raise TypeError(f"Type not serializable: {obj}") 
        
        with open(self.results_dir / f"results_{self.run_id}.jsonl", "a", encoding="utf8") as f:
            dict = {"doc": document, "content": results}
            f.write(json.dumps(dict, ensure_ascii=False, default=serialize_datetime))
            f.write("\n")
            
    def save_junk_labels(self, results):
        junk_labels = self.extract_junk_labels(results)
        self.label_counts.update(junk_labels)
        
        with open(self.results_dir / f"label_vocab_{self.run_id}.tsv", "w", encoding="utf8") as f:
            for item in self.label_counts.most_common():
                f.write(f"{item[0]}\t{item[1]}\n")
                
    def code_to_language(self):
        language_dict = {
            # EU Official Languages
            "Bulgarian": ["bul_Cyrl"],
            "Czech": ["ces_Latn"],
            "Danish": ["dan_Latn"],
            "German": ["deu_Latn"],
            "Greek": ["ell_Grek"],
            "English": ["eng_Latn"],
            "Estonian": ["est_Latn", "ekk_Latn"],
            "Finnish": ["fin_Latn"],
            "French": ["fra_Latn"],
            "Irish": ["gle_Latn"],
            "Croatian": ["hrv_Latn"],
            "Hungarian": ["hun_Latn"],
            "Italian": ["ita_Latn"],
            "Latvian": ["lav_Latn", "ltg_Latn", "lvs_Latn"],
            "Lithuanian": ["lit_Latn"],
            "Maltese": ["mlt_Latn"],
            "Dutch": ["nld_Latn"],
            "Polish": ["pol_Latn"],
            "Portuguese": ["por_Latn"],
            "Romanian": ["ron_Latn"],
            "Slovak": ["slk_Latn"],
            "Slovene": ["slv_Latn"],
            "Spanish": ["spa_Latn"],
            "Swedish": ["swe_Latn"],
            # Candidate EU Members
            "Bosnian": ["bos_Latn"],
            "Catalan": ["cat_Latn"],
            "Basque": ["eus_Latn"],
            "Galician": ["glg_Latn"],
            "Icelandic": ["isl_Latn"],
            "Georgian": ["kat_Geor"],
            "Macedonian": ["mkd_Cyrl"],
            "Albanian": ["sqi_Latn", "als_Latn"],
            "Serbian": ["srp_Cyrl", "srp_Latn"],
            "Turkish": ["tur_Latn"],
            "Ukrainian": ["ukr_Cyrl"],
            # Closely Associated Scandinavian
            "Norwegian": ["nno_Latn", "nob_Latn"],
            # Other FineTasks languages
            # Turkish and French are also FineTasks languages
            "Simplified Chinese": ["zho_Hans"],
            "Traditional Chinese": ["zho_Hant"],
            "Arabic": ["ara_Arab"],
            "Russian": ["rus_Cyrl"],
            "Thai": ["tha_Thai"],
            "Hindi": ["hin_Deva"],
            "Swahili": ["swh_Latn"],
            "Telugu": ["tel_Telu"]
        }
        
        for lang, codes in language_dict.items():
            if self.language in codes:
                return lang
            
        raise ValueError(f"Language code {self.language} is not a valid language.")

    def label_generation_pipeline(self, document):
        """This pipeline is used when the argument use_fixed labels is not given.
        This pipeline will let the LLM generate labels to lines freely, inventing new labels
        as more lines are processed. The generated labels are then compared against each other
        to find synonyms. Synonyms are combined to keep the total number of labels reasonable.
        """
        # Keep max_vocab most common labels and shuffle them. These will be given to model as options.
        vocab = [label[0] for label in self.label_counts.most_common(self.max_vocab)]
        shuffle(vocab)
        
        # Split document into even batches
        batches = self.batched(document["text"])
        
        # Put lines into a prompt and get JSON schema
        model_input = [self.format_input(input_lines=batch, vocab=vocab, task="generate_labels") for batch in batches]
        response_schema = self.get_json_schema("generate_labels")

        # Let model generate labels and invent new ones when needed
        output = self.generate(model_input, response_schema)
        
        # Combine labels that are (near) synonymous
        synonyms = self.combine_synonyms(output)
        # Format results for saving
        results = self.format_results(synonyms, batches)
        self.save_junk_labels(synonyms)
        
        # Save results
        self.save_results(document, results)
        
    def classification_pipeline(self, document):
        """This pipeline is used when the argument use_fixed_labels is given.
        This pipeline will task the LLM with classifying each line into pre-determined categories:
        either a line is 'Clean' or it is given one of eight low-quality labels."""
        # Split document into even batches
        batches = self.batched(document["text"])
        
        # Put lines into a prompt and get JSON schema
        model_input = [self.format_input(input_lines=batch, task="classify") for batch in batches]
        response_schema = self.get_json_schema("classify")

        # Classify lines according to a fixed label set.
        output = self.generate(model_input, response_schema)
        
        # Get valid labels to make sure labels are correct
        valid_labels = self.classification_labels()

        # Format results for saving
        results = self.format_results(output, batches, valid_labels=valid_labels)
        
        # Save results
        self.save_results(document, results)
        self.save_junk_labels(output)

    def process_data(self):
        logging.info("Loading data...")
        data = self.load_data()
        logging.info("Loading model...")
        self.model_setup()
        logging.info("Starting pipeline.")
        
        for idx, document in enumerate(data):
            if idx < self.start_index:
                continue
            if idx >= self.stop_index:
                break
                        
            if self.use_fixed_labels:
                self.classification_pipeline(document)
            else:
                self.label_generation_pipeline(document)

            # Keep track of progress
            if idx > 0 and idx % 1000 == 0:
                logging.info(f"Processed {idx+1} documents.")


def main():

    parser = argparse.ArgumentParser(
        description="A script for labelling line quality with LLMs."
    )

    parser.add_argument(
        "--run-id", type=str, required=True, help="ID for this run, e.g. run1"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Name of model to use.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Path to cache directory, where model is or will be saved.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="Model temperature."
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=-1,
        help="Max number of labels given in the prompt. Give -1 to use all previous labels.",
    )
    parser.add_argument(
        "--synonym-threshold",
        type=float,
        default=0.2,
        help="""Distance threshold for when two labels should count as synonyms.
        Smaller value means words are less likely to count as synonyms.""",
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Index of first document to analyse."
    )
    parser.add_argument(
        "--stop-index",
        type=int,
        default=0,
        help="Index of last document to analyse. Give -1 to analyse all documents.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Max number of lines given to the model at one time.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Path to directory, where results will be saved. Will be created, if it does not exist yet.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="eng_Latn",
        help="Language code for wanted data, eg. eng_Latn, spa_Latn."
    )
    parser.add_argument(
        "--use-fixed-labels",
        action="store_true",
        help="Use a fixed set of labels instead of generating a new label taxonomy.",
    )

    args = parser.parse_args()

    # Set the default value for --results_dir after parsing args
    if not args.results_dir:
        args.results_dir = f"../results/LLM_labelled_data/{args.run_id}"

    # Create required directories
    os.makedirs("../logs", exist_ok=True)
    os.makedirs("../results", exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Log the run settings
    with open(f"{args.results_dir}/{args.run_id}_settings.txt", "w") as f:
        f.write(f"slurm id: {os.environ.get('SLURM_JOB_ID')}\n")
        for arg, value in vars(args).items():
            logging.info(f"{arg}: {value}")
            f.write(f"{arg}: {value}\n")

    lc = LineClassifier(args)
    lc.process_data()
    logging.info("Done.")


if __name__ == "__main__":
    main()
