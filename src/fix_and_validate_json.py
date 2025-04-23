import json
import argparse
from pathlib import Path
import ast

parser = argparse.ArgumentParser()

parser.add_argument(
    "path",
    type=str
)

args = parser.parse_args()

path = Path(args.path)
checked_lines = []
with path.open("r") as f:
    print(f"Validating {path}...")
    for idx, line in enumerate(f):
        try:
            valid_line = json.loads(line, strict=False)
            
            if isinstance(valid_line["line_quality"], str):
                line_quality = valid_line["line_quality"].strip()
                valid_line["line_quality"] = ast.literal_eval(line_quality)
            checked_lines.append(valid_line)
        except Exception as e:
            print(f"LINE: {idx}: {e}; {line}")
        if len(checked_lines) == 10000:
            with open("temp.jsonl", "a") as f_out:
                f_out.writelines(f"{json.dumps(l, ensure_ascii=False)}\n" for l in checked_lines)
                checked_lines = []
    with open("temp.jsonl", "a") as f_out:
        f_out.writelines(f"{json.dumps(l, ensure_ascii=False)}\n" for l in checked_lines)
        checked_lines = []
    Path("temp.jsonl").rename(path.name)
                
         