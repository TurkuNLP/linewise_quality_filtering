import os
import json
import math
import io
import argparse
from pathlib import Path
from typing import List
from multiprocessing import Pool, cpu_count
import zstandard as zstd


def count_lines(filepath: Path, is_compressed: bool) -> int:
    """Count the number of lines in a file."""
    num_lines = 0
    read_func = yield_raw_zst if is_compressed else yield_raw_jsonl
    for _ in read_func(filepath):
        num_lines += 1
    return num_lines


def yield_raw_zst(data_path: Path):
    with open(data_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        for line in text_stream:
            if line.strip():
                yield line.strip()


def yield_raw_jsonl(data_path: Path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line.strip()


def write_chunk_to_file(chunk: List[dict], base_name: str, output_dir: Path, split_num: int):
    part_filename = f"{base_name}_{split_num:03d}.jsonl"
    part_path = output_dir / part_filename
    with open(part_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([line for line in chunk]) + '\n')


def process_jsonl_file(filepath: Path, output_dir: Path, split_count: int):
    is_compressed = filepath.suffix == '.zst'
    base_name = filepath.stem if not is_compressed else filepath.stem.split('.')[0]

    print(f"Processing {filepath.name}...", flush=True)

    total_lines = count_lines(filepath, is_compressed)
    if total_lines == 0:
        print(f"Skipping empty file: {filepath.name}", flush=True)
        return

    split_size = math.ceil(total_lines / split_count)
    read_func = yield_raw_zst if is_compressed else yield_raw_jsonl

    batch = []
    lines_in_split = 0
    split_num = 0
    for line in read_func(filepath):
        batch.append(line)
        lines_in_split += 1
        if len(batch) >= 1000 or lines_in_split >= split_size:
            write_chunk_to_file(batch, base_name, output_dir, split_num)
            batch = []
            if lines_in_split >= split_size:
                split_num += 1
                lines_in_split = 0

    if batch:
        write_chunk_to_file(batch, base_name, output_dir, split_num)
        
    print(f"Finished splitting {filepath.name} into {split_count} parts.", flush=True)


def process_jsonl_file_mp(args):
    filepath, output_dir, split_count = args
    process_jsonl_file(filepath, output_dir, split_count)


def split_jsonl_files(input_dir: str, output_dir: str, split_count: int):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jobs = []
    for file in input_path.iterdir():
        if file.name.endswith('.jsonl') or file.name.endswith('.jsonl.zst'):
            jobs.append((file, output_path, split_count))

    if not jobs:
        print("No .jsonl or .jsonl.zst files found.")
        return

    print(f"Splitting {len(jobs)} file(s) using {min(cpu_count(), len(jobs))} workers...", flush=True)

    with Pool(processes=min(cpu_count(), len(jobs))) as pool:
        pool.map(process_jsonl_file_mp, jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split large JSONL or JSONL.ZST files into smaller parts.")
    parser.add_argument('--input-dir', required=True, help='Directory containing the input JSONL files.')
    parser.add_argument('--output-dir', required=True, help='Directory to save the split JSONL files.')
    parser.add_argument('--split-count', type=int, default=10, help='Number of parts to split each file into.')

    args = parser.parse_args()

    split_jsonl_files(args.input_dir, args.output_dir, args.split_count)
