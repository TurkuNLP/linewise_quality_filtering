from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from accelerate import Accelerator #type:ignore
from datasets import load_dataset, load_from_disk, IterableDataset #type:ignore
import torch #type:ignore
from torch.nn.functional import softmax #type:ignore
from torch.utils.data import DataLoader, Dataset #type:ignore
import numpy as np
from typing import List
import argparse
import time
import json
from pathlib import Path
import zstandard as zstd #type:ignore
import io
import os

def read_zst_files(data_path):
    def yield_zst():
        with open(data_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8', errors='ignore')
            for line in text_stream:
                yield json.loads(line)

    return IterableDataset.from_generator(lambda: yield_zst())

def read_jsonl_files(data_path):
    def yield_jsonl():
        with open(data_path, "r", encoding='utf-8', errors='ignore') as f:
            for line in f:
                yield json.loads(line)

    return IterableDataset.from_generator(lambda: yield_jsonl())

def classify_lines(docs, tokenizer, model, line_batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Create list of lists, where each list is lines in doc
    lines = [doc["text"].split("\n") for doc in docs]
    # Keep track of how many lines each doc contains
    num_lines_in_each_doc = [len(doc) for doc in lines]
    # Flatten nested list
    text_batch = [item for sublist in lines for item in sublist]
    
    # Create a list of tuples (line_idx, line) to keep track of original line indices
    all_lines_with_idx = []
    for line_idx, line in enumerate(text_batch):
        all_lines_with_idx.append((line_idx, line))

    # Sort lines by their length (number of words) for efficient padding
    all_lines_with_idx.sort(key=lambda x: len(x[1]))
    
    # Identify "Clean" index
    label_map = model.config.id2label
    clean_index = {v: k for k, v in label_map.items()}["Clean"]
    
    # Here's where we will store the predictions from the minibatches
    predictions = []
    
    for i in range(0, len(all_lines_with_idx), line_batch_size):
        batch = all_lines_with_idx[i : i + line_batch_size]
    
        inputs = tokenizer(
                [tup[1] for tup in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                ).to(device)
        
        # Push batch through model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = softmax(logits, dim=-1)

            # Get predicted labels (index of max prob)
            preds = torch.argmax(probs, dim=-1)

            # Get the "Clean" score for each item
            clean_scores = probs[:, clean_index]

            for pred, score in zip(preds.cpu().tolist(), clean_scores.cpu().tolist()):
                predictions.append({
                    "label": label_map[pred],
                    "clean_score": round(score, 3)
                    })

    # Extract labels and scores with line idx
    labels_with_idx = [(tup[0], pred["label"]) for tup, pred in zip(all_lines_with_idx, predictions)]
    scores_with_idx = [(tup[0], pred["clean_score"]) for tup, pred in zip(all_lines_with_idx, predictions)]
    
    # Sort back into original order
    labels_with_idx.sort(key=lambda x: x[0])
    scores_with_idx.sort(key=lambda x: x[0])
    
    # Drop idx
    labels = [tup[1] for tup in labels_with_idx]
    scores = [tup[1] for tup in scores_with_idx]
    
    results = []
    start_index = 0
    end_index = 0
    for idx, num_lines in enumerate(num_lines_in_each_doc):
        end_index += num_lines
        d = {
            "text": docs[idx]["text"],
            "id": docs[idx]["id"],
            "langs": docs[idx]["seg_langs"],
            "robotstxt": docs[idx]["robotstxt"],
            "collection": docs[idx]["collection"],
            "url": docs[idx]["u"],
            "document_lang": docs[idx]["lang"][np.argmax(docs[idx]["prob"])],
            "line_quality_labels": labels[start_index:end_index],
            "quality_scores": scores[start_index:end_index],
        }

        assert len(d["text"].split("\n")) == len(d["line_quality_labels"]) == len(d["quality_scores"])
        results.append(d)
        start_index += num_lines

    return results

def save_results(results, save_path):
    with open(save_path, "a") as f:
        for doc in results:
            if doc["text"] and doc["robotstxt"] == "allowed": # Save only non-empty and robots.txt compliant docs
                f.write(f"{json.dumps(doc, ensure_ascii=False)}\n")


def file_exists_and_line_count(path):
    if not os.path.isfile(path):
        return False, 0

    with open(path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    return True, line_count

def filter_row(row, idx_to_keep):
    text_lines = row["text"].split("\n")
    quality_labels = row["line_quality_labels"]
    quality_scores = row["quality_scores"]
    line_langs = row["langs"]
    
    assert len(text_lines) == len(quality_labels) == len(quality_scores)
    
    filtered_lines = [text_lines[i] for i in idx_to_keep]
    row["text"] = "\n".join(filtered_lines)
    row["line_quality_labels"] = [quality_labels[i] for i in idx_to_keep]
    row["quality_scores"] = [quality_scores[i] for i in idx_to_keep]
    row["langs"] = [line_langs[i] for i in idx_to_keep]
    
    return row

def trim_row(row, start, end):
    text_lines = row['text'].split("\n")
    quality_labels = row['line_quality_labels']
    quality_scores = row["quality_scores"]
    # Slice only the retained portion
    if start <= end:
        trimmed_lines = text_lines[start:end+1]
        trimmed_quality_labels = quality_labels[start:end+1]
        trimmed_quality_scores = quality_scores[start:end+1]
    else:
        trimmed_lines = []  # all lines are low quality
        trimmed_quality_labels = []
        trimmed_quality_scores = []
    
    row["text"] = "\n".join(trimmed_lines)
    row["line_quality_labels"] = trimmed_quality_labels
    row["quality_scores"] = trimmed_quality_scores
    
    return row

def remove_lines(row, filter=False, trim=False):
    quality_labels = row["line_quality_labels"]

    if filter and trim:
        raise ValueError("Only one of 'trim' or 'filter' can be provided, not both.")

    #Remove all lines with given quality labels.
    elif filter:
        # Get idx of lines to keep
        clean_idx = [idx for idx, quality_label in enumerate(quality_labels) if quality_label == "Clean" and row["quality_scores"][idx] > 0.9]
        idx_to_keep = [idx for idx in clean_idx if row["langs"][idx] == row["document_lang"]]
        row = filter_row(row, idx_to_keep)
        return row

    #Remove only lines from start and end of document that have given quality labels
    elif trim:
        # Find start index: first index where line label is not in given quality labels
        start = 0
        while start < len(quality_labels) and quality_labels[start] != "Clean":
            start += 1

        # Find end index: last index where line label is not in given quality labels
        end = len(quality_labels) - 1
        while end >= 0 and quality_labels[end] != "Clean":
            end -= 1
            
        row = trim_row(row, start, end)
        return row
    
    else:
        raise ValueError("Must choose either 'trim' or 'filter'")
    
def process_batch(args, docs, batch_num, tokenizer, model):
    if tokenizer and model:
        classified = classify_lines(docs, tokenizer, model, args.line_batch_size)
    else:
        # Skip classification. Data should be classified and contain the field "line_quality_labels" and "quality_scores"
        classified = docs

    # Trim or filter documents
    filtered = [remove_lines(doc, filter=args.filter, trim=args.trim) for doc in classified]

    # After filtering/trimming, keep only docs that are above the minimum length
    processed = [doc for doc in filtered if len(doc["text"]) >= args.min_doc_length]

    save_results(processed, args.save_path)
    print(f"Processed and saved batch {batch_num}, total: {(batch_num+1)*args.doc_batch_size} documents", flush=True)
    print(f"Salvaged {len(processed)} documents", flush=True)

def main(args, tokenizer, model):
    if args.data_path.endswith(".zst"):
        data = read_zst_files(args.data_path)
    elif args.data_path.endswith(".jsonl"):
        data = read_jsonl_files(args.data_path)

    exists, lines = file_exists_and_line_count(args.save_path)
    if exists:
        print(f"File {args.save_path} already exists with {lines} lines. "
              f"Starting from line {lines}.")
        start_index = lines
    else:
        print(f"File {args.save_path} does not exist. Starting from line 0.")
        start_index = 0
    
    batch_num = 0
    docs = []
    start = time.time()
    for doc_index, doc in enumerate(data):
        if doc_index < start_index:
            continue
        if doc["doc_scores"][0] < 5 and len(doc["text"]) >= args.min_doc_length:
            docs.append(doc)
        if len(docs) == args.doc_batch_size:
            # TODO: check segment langs before classifying to save compute
            process_batch(args, docs, batch_num, tokenizer, model)
            docs = []
            batch_num += 1
            end = time.time()
            print(f"Processing {args.doc_batch_size} docs took {end - start:.2f} seconds.", flush=True)
            start = time.time()
            
    # Process any remaining documents
    if docs:
        process_batch(args, docs, batch_num, tokenizer, model)
        docs = []
        batch_num += 1
        end = time.time()
        print(f"Processing {args.doc_batch_size} docs took {end - start:.2f} seconds.", flush=True)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for filtering low-quality segments.")
    parser.add_argument("--model-path", type=str, default=None) # Path to classifier, if needed
    parser.add_argument("--classify", action="store_true") # Give this argument, if the data is not classified yet
    parser.add_argument("--data-path", type=str, required=True) # Path to data (jsonl or jsonl.zst)
    parser.add_argument("--save-path", type=str, required=True) # Path to where results are saved
    parser.add_argument("--doc-batch-size", type=int, default=128) # How many documents are processed at once
    parser.add_argument("--line-batch-size", type=int, default=128) # How many lines are processed at once
    parser.add_argument("--min-doc-length", type=int, default=250) # After filtering/trimming, docs with fewer chars will be discarded
    parser.add_argument("--trim", action="store_true") # Trim: remove unclean lines from beginning and end until Clean line is encountered
    parser.add_argument("--filter", action="store_true") # Filter: remove all unclean lines
    args = parser.parse_args()

    if args.classify:
        if not args.model_path:
            raise ValueError("Model path must be provided for classification.")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path,
                                                                torch_dtype=torch.float16,
                                                                device_map="auto",
                                                                local_files_only=True
                                                                )
        model.half()
        model.eval()
        
        print(f"{torch.cuda.device_count()} GPUs available.", flush=True)
        print("Model on device:", model.device, flush=True)
        
    if not args.classify:
        tokenizer = None
        model = None
        print("Skipping classification. Assuming that data is pre-classified "
              "and contains the fields 'line_quality_labels' and 'quality_scores'. "
              "Use --classify to enable classification.",
              flush=True)
        if args.model_path:
            print("Ignoring provided model path since --classify is not given.", flush=True)
        if torch.cuda.is_available():
            print("GPU available but not used. You should only reserve GPUs when --classify is given.", flush=True)

    main(args, tokenizer, model)
    
    print("Done.", flush=True)