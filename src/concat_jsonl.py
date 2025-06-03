import argparse
from pathlib import Path

def concatenate_jsonl_files(input_dir, output_file):
    input_path = Path(input_dir)
    jsonl_files = sorted(input_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found in directory: {input_dir}")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure the output directory exists
    
    with output_path.open('w', encoding='utf-8') as fout:
        for file_path in jsonl_files:
            print(f"Adding {file_path}...")
            with file_path.open('r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)

    print(f"\nDone. {len(jsonl_files)} files merged into: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Concatenate all JSONL files in a directory.')
    parser.add_argument(
        '--input-dir', 
        required=True, 
        help='Directory containing .jsonl files to merge'
    )
    parser.add_argument(
        '--output', 
        required=True, 
        help='Output .jsonl file path (e.g., merged.jsonl)'
    )
    args = parser.parse_args()
    
    concatenate_jsonl_files(args.input_dir, args.output)

if __name__ == '__main__':
    main()
