from datasets import load_dataset #type:ignore
import json
import argparse
from pathlib import Path
import ast
import os

def load_data(path):
    dataset = load_dataset(
        "parquet",
        data_files= {"train": path},
        streaming=True 
    )
    
    return dataset["train"]

def remove_lines(row, quality_threshold, filter=False, trim=False):
    text_lines = row["text"].splitlines()
    line_quality = row["line_quality"]
    
    if isinstance(line_quality, str):
        line_quality = line_quality.strip()
        line_quality = ast.literal_eval(line_quality)
        
    if filter and trim:
        raise ValueError("Only one of 'trim' or 'filter' can be provided, not both.")
    
    elif filter:
        # Filter lines based on the quality threshold
        filtered_lines = [line for line, quality in zip(text_lines, line_quality) if quality >= quality_threshold]

        # Rebuild the text by joining the filtered lines with newline characters
        row["line_quality"] = [line for line in line_quality if line >= quality_threshold]
        row["text"] = "\n".join(filtered_lines)
        return row
    
    elif trim:
        # Find start index: first index where quality >= threshold
        start = 0
        while start < len(line_quality) and line_quality[start] < quality_threshold:
            start += 1

        # Find end index: last index where quality >= threshold
        end = len(line_quality) - 1
        while end >= 0 and line_quality[end] < quality_threshold:
            end -= 1

        # Slice only the retained portion
        if start <= end:
            filtered_lines = text_lines[start:end+1]
        else:
            filtered_lines = []  # all lines are low quality

        row["line_quality"] = [line for line in line_quality if line >= quality_threshold]
        row["text"] = "\n".join(filtered_lines)
        return row
    
    else:
        raise ValueError("Must choose either 'trim' or 'filter'")

def file_exists_and_line_count(path):
    if not os.path.isfile(path):
        return False, 0

    with open(path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    return True, line_count

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quality-threshold", type=float, default=0.9,
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
    
    with open(args.save_path, "a", encoding="utf-8") as f_out:
        batched_results = []
        for idx, row in enumerate(data):
            if idx < start_index:
                continue
            processed = remove_lines(row, args.quality_threshold, filter=args.filter, trim=args.trim)
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