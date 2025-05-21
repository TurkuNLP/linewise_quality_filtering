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

def collate_fn(batch):
    tokenized = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    return tokenized

def accelerate_multi_gpu_inference(texts: List[str], batch_size: int) -> List[int]:
    dataset = TextDataset(texts)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        persistent_workers=False,
        num_workers=0
    )
    dataloader = accelerator.prepare(dataloader)  # Multi-GPU ready
    all_preds = []

    # Get class index for label "Clean"
    class_index_clean = {v: k for k, v in label_map.items()}["Clean"]
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            gathered_preds = accelerator.gather(preds)
            gathered_probs = accelerator.gather(probs)
            
            for pred, prob in zip(gathered_preds.cpu().tolist(), gathered_probs.cpu().tolist()):
                all_preds.append({
                    "label": label_map[pred],
                    "clean_score": round(prob[class_index_clean], 3)
                })
            
            # Memory clean up
            del batch, outputs, logits, probs, preds, gathered_preds, gathered_probs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    return all_preds

def read_zst_files(data_path):
    def yield_zst():
        with open(data_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8',errors='ignore') 
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

def process_batch(docs):
    # Create list of lists, where each list is lines in doc
    lines = [doc["text"].split("\n") for doc in docs]
    # Keep track of how many lines each doc contains
    num_lines_in_each_doc = [len(doc) for doc in lines]
    # Flatten nested list
    text_batch = [item for sublist in lines for item in sublist]
    
    all_lines_with_idx = []
    for line_idx, line in enumerate(text_batch):
        all_lines_with_idx.append((line_idx, line))
        
    # Sort lines by their length (number of words) for efficient padding
    all_lines_with_idx.sort(key=lambda x: len(x[1]))

    # Predict
    predictions = accelerate_multi_gpu_inference([tup[1] for tup in all_lines_with_idx], batch_size=args.line_batch_size)
    
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
            "quality_score": scores[start_index:end_index]
        }
        # Ensure the lines, labels and scores match
        assert len(d["text"].split("\n")) == len(d["line_quality_labels"]) == len(d["quality_score"])
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
def main(args):
    #data = load_data(args.language, source=args.data_source)
    print(f"Loading data from {args.data_path}", flush=True)
    if args.data_path.endswith(".jsonl"):
        data = read_jsonl(args.data_path)
    elif args.data_path.endswith(".zst"):
        print("Data is compressed with zstandard. Decompressing...", flush=True)
        data = read_zst_files(args.data_path)
    else:
        raise ValueError(f"Unsupported file format: {args.data_path}.")
    exists, lines = file_exists_and_line_count(args.save_path)
    if exists:
        print(f"File {args.save_path} already exists with {lines} lines. "
              f"Starting from line {lines}.", flush=True)
        start_index = lines
    else:
        print(f"File {args.save_path} does not exist. Starting from line 0.", flush=True)
        start_index = 0
    docs = []
    batch_num = 0
    for doc_index, doc in enumerate(data):
        if doc_index < start_index:
            continue
        docs.append(doc)
        if len(docs) == args.doc_batch_size:
            results = process_batch(docs)
            save_results(results, args.save_path)
            docs = []
            batch_num += 1
    
    # Process any remaining documents
    if docs:
        results = process_batch(docs)
        save_results(results, args.save_path)
        docs = []
        batch_num += 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script for labelling line quality with finetuned classifier."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to finetuned model.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="french",
        help="Data language. Make sure your model is trained for this language."
    )
    parser.add_argument(
        "--data-source",
        type=str,
        help="Where to get the data from. Give either huggingface, local_hplt or local_fineweb."
    )
    parser.add_argument(
        "--doc-batch-size",
        type=int,
        default=128,
        help="Number of documents to process at once."
    )
    parser.add_argument(
        "--line-batch-size",
        type=int,
        default=512,
        help="Number of lines to process at once."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True
    )
    args = parser.parse_args()
    
    # === Accelerator setup ===
    accelerator = Accelerator()
    device = accelerator.device

    # === Load model/tokenizer ===

    MODEL_NAME = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        local_files_only=True,
        )
    label_map = model.config.id2label
    model = accelerator.prepare(model)  # Prepares for multi-GPU
    model.eval()

    main(args)

