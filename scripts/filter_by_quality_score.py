from datasets import load_dataset
import json

def load_data():
    return load_dataset(
        "TurkuNLP/finerweb-10bt",
        split="train",
        streaming=True)


def preprocess_row(row, quality_threshold):
    text_lines = row['text'].splitlines()
    line_quality = row['line_quality']

    # Filter lines based on the quality threshold
    filtered_lines = [line for line, quality in zip(text_lines, line_quality) if quality >= quality_threshold]

    # Rebuild the text by joining the filtered lines with newline characters
    new_text = "\n".join(filtered_lines)

    # Replace the 'text' field with the new preprocessed text
    row['text'] = new_text
    return row


if __name__ == "__main__":
    data = load_data()
    with open("../data/finerweb-10BT-threshold-09.jsonl", "w", encoding="utf-8") as f_out:
        for idx, row in enumerate(data):
            filtered = preprocess_row(row, 0.9)
            f_out.write(f"{json.dumps(filtered, ensure_ascii=False)}\n")