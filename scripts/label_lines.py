# Standard libraries
import argparse
from collections import Counter
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
import pandas as pd  # type: ignore
from pydantic import BaseModel, RootModel  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
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
        self.run_id = args.run_id
        self.cache_dir = args.cache_dir
        self.model_name = args.model_name
        self.temperature = args.temperature
        self.max_vocab = args.max_vocab
        self.synonym_threshold = args.synonym_threshold
        self.start_index = args.start_index
        self.stop_index = args.stop_index
        self.batch_size = args.batch_size
        self.use_previous_labels = args.use_previous_labels
        self.results_dir = args.results_dir

    def model_setup(self):
        self.model = LLM(
            model=self.model_name,
            download_dir=self.cache_dir,
            dtype="bfloat16",
            max_model_len=128_000,
            tensor_parallel_size=torch.cuda.device_count(),
            # pipeline_parallel_size=2, # use if multiple nodes are needed
            enforce_eager=False,
            gpu_memory_utilization=0.8,
        )

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

        validated_outputs = []
        for output in outputs:
            if not self.validate_json(output):
                output = self.generate(output, response_schema)
            validated_outputs.append(output)

        return validated_outputs

    def format_input(
        self, input_lines=None, vocab=None, group_name=None, synonyms=None, task=None
    ):
        """Format input lines."""

        if task == "classify":
            formatted_lines = ""
            for i, line in enumerate(input_lines):
                formatted_lines += f"*Line {i+1}:* {line}\n------\n"

            if len(vocab) == 0:
                vocab = (
                    "The list is currently empty. You are free to create new labels."
                )

            prompt = prompts.line_quality_prompt(formatted_lines, vocab)
            return prompt

        if task == "synonyms":
            prompt = prompts.review_synonyms(group_name, synonyms)
            return prompt

    def batched(self, doc):
        lines = doc.strip().split("\n")
        total_lines = len(lines)

        # Determine the number of batches
        num_batches = math.ceil(total_lines / self.batch_size)

        # If not evenly divisible, adjust batch sizes to be as even as possible
        batch_size = math.ceil(total_lines / num_batches)

        # Split into batches
        batches = [lines[i : i + batch_size] for i in range(0, total_lines, batch_size)]

        return batches

    def validate_json(self, model_output):
        try:
            json.loads(model_output, strict=False)
            return True
        except json.JSONDecodeError as e:
            logging.debug(e)
            logging.debug("Invalid JSON output:")
            logging.debug(repr(model_output))
        return False

    def extract_junk_labels(self, output):
        """Extract junk labels from output and add them to the junk list."""
        labels = [label for label in output.values()]

        junk_labels = []
        for label in labels:
            label = label.lower().strip()
            if label != "clean":
                junk_labels.append(label)

        junk_labels = list(set(junk_labels))

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

        if task == "classify":

            class ResponseFormat(BaseModel):
                root: dict[str, str]

        elif task == "synonyms":

            class ResponseFormat(RootModel):
                root: dict[str, list[str]]

        json_schema = ResponseFormat.model_json_schema()
        return GuidedDecodingParams(json=json_schema)

    def load_data(self):
        return load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")

    def load_previous_labels(self):
        junk_labels = Counter()
        try:
            with open(
                self.result_dir / f"descriptor_vocab_{self.run_id}.tsv", "r"
            ) as f:
                file = f.readlines()
                for line in file:
                    line = line.strip().split("\t")
                    label, freq = line[0], int(line[1])
                    junk_labels[label] += freq
            return junk_labels
        except FileNotFoundError:
            return junk_labels

    def find_synonym_candidates(self, labels, embeddings):
        # Convert embeddings to NumPy array if needed
        embeddings = np.array(embeddings)

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
        labels = clustering.fit_predict(embeddings)

        # Group words by cluster labels
        groups = {}
        for idx, label in enumerate(labels):
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

    def combine_synonyms(self, labels):

        junk_labels = self.extract_output(labels)

        try:
            with open(
                self.result_dir / f"descriptor_vocab_{self.run_id}.tsv", "r"
            ) as f:
                file = f.readlines()
                junk_labels = [line.split("\t")[0] for line in file] + junk_labels
        except FileNotFoundError:
            pass

        # Embed best descriptors
        embedder = StellaEmbedder()
        embeddings = embedder.embed_descriptors(junk_labels)

        # Group similar descriptors
        synonym_candidates = self.find_synonym_candidates(junk_labels, embeddings)

        # Use LLM to evaluate and form final synonyms
        prompts = [
            self.format_prompt(task="synonyms", group_name=group_name, synonyms=syns)
            for group_name, syns in synonym_candidates.items()
        ]

        json_schema = self.get_response_format(task="synonyms")
        synonym_groups = self.generate(prompts, json_schema)

        labels = self.replace_synonyms(synonym_groups, labels)

        return labels

    def replace_synonyms(self, synonyms, results):
        # Create a mapping from synonym to its group for fast lookup
        synonym_map = {
            syn: group for group, members in synonyms.items() for syn in members
        }

        # Replace synonyms
        for doc in results.values():
            replaced = [synonym_map.get(label, label) for label in doc.values()]

    def save_results(self, results):
        with open(self.results_dir / f"results_{self.run_id}.json", "w") as f:
            json.dump(results, f)

    def process_data(self):
        self.model_setup()
        data = self.load_data()
        if self.use_previous_labels:
            vocab = self.load_previous_labels("junk_labels.tsv")
        else:
            vocab = Counter()

        for idx, doc in enumerate(data["text"]):
            if idx < self.start_index:
                continue
            start_time = time.time()
            for batches in self.batched(doc):
                model_input = [self.format_input(batch, vocab) for batch in batches]
                response_schema = self.get_json_schema("classify")
                output = self.generate(model_input, response_schema)
                results = self.combine_synonyms(output)
                self.save_results(results)
            end_time = time.time()
            logging.info(f"Time taken for document {idx}: {end_time - start_time}")
            if idx > self.stop_index:
                break


def main():

    parser = argparse.ArgumentParser(
        description="A script for labelling line quality with LLMs."
    )

    parser.add_argument(
        "--run-id", type=str, required=True, help="ID for this run, e.g. run1"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="../.cache",
        help="Path to cache directory, where model is or will be saved.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Name of model to use.",
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
        "--use-previous-labels",
        action="store_true",
        help="Use descriptors used in a previous run as a starting point.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Path to directory, where results will be saved. Will be created, if it does not exist yet.",
    )

    args = parser.parse_args()

    # Set the default value for --results_dir after parsing args
    if not args.results_dir:
        args.results_dir = f"../results/{args.run_id}"

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
