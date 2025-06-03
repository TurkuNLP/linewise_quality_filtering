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

def read_jsonl(data_path):
    def yield_jsonl():
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    yield json.loads(line)

    return IterableDataset.from_generator(lambda: yield_jsonl())

def process_batch(docs, tokenizer, model, line_batch_size):
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
            "line_quality_labels": labels[start_index:end_index],
            "quality_scores": scores[start_index:end_index]
        }
        assert len(d["text"].split("\n")) == len(d["line_quality_labels"]) == len(d["quality_scores"])
        results.append(d)
        start_index += num_lines

    return results
        

def save_results(results, save_path):
    with open(save_path, "a") as f:
        for doc in results:
            f.write(f"{json.dumps(doc, ensure_ascii=False)}\n")
            

def file_exists_and_line_count(path):
    if not os.path.isfile(path):
        return False, 0

    with open(path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    return True, line_count
    

# === Example ===
def main(args, tokenizer, model):
    print(f"Loading data from {args.data_path}", flush=True)
    if args.data_path.endswith(".jsonl"):
        data = read_jsonl(args.data_path)
    elif args.data_path.endswith(".zst"):
        print("Data is compressed with zstandard. Decompressing...", flush=True)
        data = read_zst_files(args.data_path)
    else:
        raise ValueError(f"Unsupported file format: {args.data_path}.")
    
    if args.save_path.endswith(".zst"):
        args.save_path = args.save_path.removesuffix(".zst")
    
    exists, lines = file_exists_and_line_count(args.save_path)
    if exists:
        print(f"File {args.save_path} already exists with {lines} lines. "
              f"Starting from line {lines}.")
        start_index = lines
    else:
        print(f"File {args.save_path} does not exist. Starting from line 0.")
        start_index = 0
    docs = []
    batch_num = 0
    start = time.time()
    for doc_index, doc in enumerate(data):
        if doc_index < start_index:
            continue
        docs.append(doc)
        if len(docs) == args.doc_batch_size:
            results = process_batch(docs, tokenizer, model, args.line_batch_size)
            save_results(results, args.save_path)
            print(f"Processed and saved batch {batch_num}, total: {(batch_num+1)*args.doc_batch_size} documents", flush=True)
            docs = []
            batch_num += 1
            end = time.time()
            print(f"Processing {args.doc_batch_size} took {end - start:.2f} seconds.", flush=True)
            start = time.time()
    
    # Process any remaining documents
    if docs:
        results = process_batch(docs, tokenizer, model, args.line_batch_size)
        save_results(results, args.save_path)
        print(f"Processed and saved batch {batch_num}", flush=True)
        docs = []
        batch_num += 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for labelling line quality with finetuned classifier.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--doc-batch-size", type=int, default=128)
    parser.add_argument("--line-batch-size", type=int, default=128)
    args = parser.parse_args()
    
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
        
    main(args, tokenizer, model)