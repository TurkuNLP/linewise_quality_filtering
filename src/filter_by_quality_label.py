from datasets import load_dataset #type:ignore
import json
import argparse
from pathlib import Path
import ast
import os

def load_data(path):
    dataset = load_dataset(
        "json",
        data_files= {"train": path},
        streaming=True 
    )
    
    return dataset["train"]

def filter_row(row, idx_to_keep):
    text_lines = row["text"].splitlines()
    quality_labels = row["line_quality_labels"]
    quality_scores = row["quality_score"]
    
    assert len(text_lines) == len(quality_labels) == len(quality_scores)
    
    filtered_lines = [text_lines[i] for i in idx_to_keep]
    row["text"] = "\n".join(filtered_lines)
    row["line_quality_labels"] = [quality_labels[i] for i in idx_to_keep]
    row["quality_score"] = [quality_scores[i] for i in idx_to_keep]
    
    return row

def trim_row(row, start, end):
    text_lines = row['text'].splitlines()
    quality_labels = row['line_quality_labels']
    quality_scores = row["quality_score"]
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
    row["quality_score"] = trimmed_quality_scores
    
    return row

def remove_lines(row, labels_to_remove, filter=False, trim=False):
    quality_labels = row["line_quality_labels"]
    
    if isinstance(labels_to_remove, str):
        labels_to_remove = [labels_to_remove]
        
    if filter and trim:
        raise ValueError("Only one of 'trim' or 'filter' can be provided, not both.")

    #Remove all lines with given quality labels.
    elif filter:
        # Get idx of lines to keep
        idx_to_keep = [idx for idx, quality_label in enumerate(quality_labels) if quality_label not in labels_to_remove]
        row = filter_row(row, idx_to_keep)
        return row

    #Remove only lines from start and end of document that have given quality labels
    elif trim:
        # Find start index: first index where line label is not in given quality labels
        start = 0
        while start < len(quality_labels) and quality_labels[start] in labels_to_remove:
            start += 1

        # Find end index: last index where line label is not in given quality labels
        end = len(quality_labels) - 1
        while end >= 0 and quality_labels[end] in labels_to_remove:
            end -= 1
            
        row = trim_row(row, start, end)
        return row
    
    else:
        raise ValueError("Must choose either 'trim' or 'filter'")


def file_exists_and_line_count(path):
    if not os.path.isfile(path):
        return False, 0

    with open(path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    return True, line_count

def parse_labels(args):
    valid_quality_labels = [
        "Clean",
        "Bibliographical & Citation References",
        "Contact & Identification Information",
        "Formatting, Style & Errors",
        "Legal & Administrative Content",
        "Navigation & Interface Elements",
        "Offensive or Inappropriate Content",
        "Promotional & Spam Content",
        "Technical Specifications & Metadata"
    ]
    
    shorthand2truelabel = {
        "clean": valid_quality_labels[0],
        "citations": valid_quality_labels[1],
        "contact": valid_quality_labels[2],
        "errors": valid_quality_labels[3],
        "legal": valid_quality_labels[4],
        "interface": valid_quality_labels[5],
        "toxic": valid_quality_labels[6],
        "spam": valid_quality_labels[7],
        "tech": valid_quality_labels[8]
    }
    
    if args.quality_labels == "all":
        quality_labels = valid_quality_labels[1:]
    else:
        quality_labels = [label.strip() for label in args.quality_labels.split(",")]
        quality_labels = [shorthand2truelabel.get(label, label) for label in quality_labels]
    
    for label in quality_labels:
        if label not in valid_quality_labels:
            raise ValueError(f"Label '{label}' is not a valid quality label.")
            
    return quality_labels


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quality-labels", type=str, required=True, help="Name of labels to be removed."
        )
    parser.add_argument(
        "--filter", action="store_true", help="Filter out all lines that fall below quality threshold"
    )
    parser.add_argument(
        "--trim", action="store_true", help="Trim lines that fall below quality threshold from start and end of document" 
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
    )
    parser.add_argument(
        "--save-path", type=str, required=True,
        help="Path to save the filtered data"
    )
    
    args = parser.parse_args()
    data = load_data(args.data_path)
    
    exists, lines = file_exists_and_line_count(args.save_path)
    if exists:
        print(f"File {args.save_path} already exists with {lines} lines. "
              f"Starting from line {lines}.")
        start_index = lines
    else:
        print(f"File {args.save_path} does not exist. Starting from line 0.")
        start_index = 0
    
    labels = parse_labels(args)
    
    if args.filter:
        print(f"FILTERING LINES WITH QUALITY LABELS: {labels}")
    elif args.trim:
        print(f"TRIMMING LINES WITH QUALITY LABELS: {labels}")
    
    with open(args.save_path, "a", encoding="utf-8") as f_out:
        batched_results = []
        for idx, row in enumerate(data):
            if idx < start_index:
                continue
            processed = remove_lines(row, labels, filter=args.filter, trim=args.trim)
            # If all lines were removed, remove the document completely.
            if processed["text"]:
                batched_results.append(processed)
            # Write in batches of 1k documents to save on I/0
            if len(batched_results) == 1_000:
                f_out.writelines(f"{json.dumps(result, ensure_ascii=False)}\n" for result in batched_results)
                batched_results = []
        
        # Write any remaining documents
        if batched_results:
            f_out.writelines(f"{json.dumps(result, ensure_ascii=False)}\n" for result in batched_results)
            batched_results = []