import json
from pathlib import Path
import random
import pandas as pd #type:ignore

sample_indices = random.sample(range(100_001), 100)

# Choose files and get their unique identifies to find the same files in other documents
file = Path("..")/"data"/f"finerweb-350BT-trimmed-threshold-09"/"001_00000.jsonl"

identifiers = []
with open(file, "r") as f:
    for idx, line in enumerate(f):
        if idx in sample_indices:
            line = json.loads(line, strict=False)
            identifiers.append(line["id"])

for variant in ["filtered-threshold-05", "filtered-threshold-09", "trimmed-threshold-05", "trimmed-threshold-09"]:
    file = Path("..")/"data"/f"finerweb-350BT-{variant}"/"001_00000.jsonl"
    lines = []
    with file.open("r") as f:
        for line in f:
            line = json.loads(line, strict=False)
            if line["id"] in identifiers:
                lines.append(line)
                
    with open(f"{variant}.jsonl", "w") as f_out:
        for line in lines:
            f_out.write(f"{json.dumps(line, ensure_ascii=False)}\n")
        
        
og_path = Path("/scratch/project_462000615/ehenriks/FINEWEB/quality_predictions/sample/350BT")
df = pd.read_parquet(og_path/"001_00000.parquet")

# Filter rows by index
filtered_df = df[df['id'].isin(identifiers)]

filtered_df.to_json("fineweb_sample.jsonl", orient="records", lines=True)

