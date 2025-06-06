from transformers import AutoTokenizer, AutoModelForSequenceClassification
from accelerate import Accelerator #type:ignore
from datasets import load_dataset, load_from_disk, IterableDataset #type:ignore
import torch #type:ignore
from torch.nn.functional import softmax #type:ignore
from torch.utils.data import DataLoader, Dataset #type:ignore
from typing import List
import argparse
import time
import json
from pathlib import Path
import zstandard as zstd #type:ignore
import io
import os


# === Dataset ===
class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def get_collate_fn(tokenizer):
    def collate(batch):
        return tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
    return collate

def accelerate_multi_gpu_inference(texts: List[str], batch_size: int, model, tokenizer, accelerator, label_map: dict) -> List[int]:
    dataset = TextDataset(texts)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=get_collate_fn(tokenizer),
        persistent_workers=False,
    )
    dataloader = accelerator.prepare(dataloader)
    all_preds = []

    device = accelerator.device
    class_index_clean = {v: k for k, v in label_map.items()}["Clean"]

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            gathered_preds = accelerator.gather(preds)
            gathered_probs = accelerator.gather(probs)

            for pred, prob in zip(gathered_preds.cpu().tolist(), gathered_probs.cpu().tolist()):
                all_preds.append({
                    "label": label_map[pred],
                    "clean_score": round(prob[class_index_clean], 3)
                })

            #del batch, outputs, probs, preds, gathered_preds, gathered_probs
            #if torch.cuda.is_available():
            #    torch.cuda.empty_cache()

    return all_preds

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
                if line.strip():
                    yield json.loads(line)

    return IterableDataset.from_generator(lambda: yield_jsonl())

def process_batch(docs, model, tokenizer, accelerator, label_map, args):
    lines = [doc["text"].split("\n") for doc in docs]
    num_lines_in_each_doc = [len(doc) for doc in lines]
    text_batch = [item for sublist in lines for item in sublist]

    all_lines_with_idx = [(i, l) for i, l in enumerate(text_batch)]
    all_lines_with_idx.sort(key=lambda x: len(x[1]))

    predictions = accelerate_multi_gpu_inference(
        [t[1] for t in all_lines_with_idx],
        batch_size=args.line_batch_size,
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        label_map=label_map
    )

    labels_with_idx = [(t[0], p["label"]) for t, p in zip(all_lines_with_idx, predictions)]
    scores_with_idx = [(t[0], p["clean_score"]) for t, p in zip(all_lines_with_idx, predictions)]

    labels_with_idx.sort(key=lambda x: x[0])
    scores_with_idx.sort(key=lambda x: x[0])

    labels = [t[1] for t in labels_with_idx]
    scores = [t[1] for t in scores_with_idx]

    results = []
    start_index = 0
    end_index = 0
    for idx, num_lines in enumerate(num_lines_in_each_doc):
        end_index += num_lines
        d = {
            "text": docs[idx]["text"],
            "id": docs[idx]["id"],
            "line_quality_labels": labels[start_index:end_index],
            "quality_score": scores[start_index:end_index]
        }
        assert len(d["text"].split("\n")) == len(d["line_quality_labels"]) == len(d["quality_score"])
        results.append(d)
        start_index += num_lines
        
    # Empty the GPU cache to avoid memory issues
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Clean up variables to free memory
    del lines, text_batch, all_lines_with_idx, labels_with_idx, scores_with_idx, labels, scores

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

def main(args):
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
    start_index = lines if exists else 0
    print(f"Starting from line {start_index}.", flush=True)

    accelerator = Accelerator()
    device = accelerator.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        device_map="auto",
    )
    label_map = model.config.id2label
    model = accelerator.prepare(model)
    model.eval()
    
    print(f"{torch.cuda.device_count()} GPUs available.", flush=True)
    print("Model on device:", device, flush=True)

    docs = []
    batch_num = 0
    for doc_index, doc in enumerate(data):
        if doc_index < start_index:
            continue
        docs.append(doc)
        if len(docs) == args.doc_batch_size:
            print(f"Processing batch {batch_num + 1}...", flush=True)
            results = process_batch(docs, model, tokenizer, accelerator, label_map, args)
            print(f"Batch {batch_num + 1} processed.", flush=True)
            print(f"Saving results to {args.save_path}...", flush=True)
            save_results(results, args.save_path)
            docs = []
            batch_num += 1

    if docs:
        results = process_batch(docs, model, tokenizer, accelerator, label_map, args)
        save_results(results, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for labelling line quality with finetuned classifier.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--language", type=str, default="french")
    parser.add_argument("--data-source", type=str)
    parser.add_argument("--doc-batch-size", type=int, default=128)
    parser.add_argument("--line-batch-size", type=int, default=512)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()
    main(args)


