import pandas as pd #type:ignore
import json
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
from collections import Counter
import argparse
import random

def read_data(path):
    data = []
    with open(path) as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def create_dataframe(data):
    texts, labels = [], []
    for document in data:
        for line in document["content"]:
            texts.append(line["line"])
            labels.append(line["label"])
    df = pd.DataFrame()
    df["text"] = texts
    df["label"] = labels
    return df


def stratified_split_and_encode(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    train_size: float = 0.7,
    random_state: int = 42,
):
    assert train_size < 1
    
    # 1. Stratified split
    train_df, temp_df = train_test_split(
        df,
        test_size=1-train_size,
        stratify=df[label_column],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df[label_column],
        random_state=random_state,
    )

    # 2. Build custom label -> ID mapping
    label_to_zero = "Clean"
    unique_labels = df[label_column].unique().tolist()
    other_labels = [label for label in unique_labels if label != label_to_zero]
    ordered_labels = [label_to_zero] + sorted(other_labels)

    label2id = {label: idx for idx, label in enumerate(ordered_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    # 3. Encode labels and reset index
    def encode_and_clean(df):
        df = df.copy()
        df[label_column] = df[label_column].map(label2id)
        return df.reset_index(drop=True)

    train_df = encode_and_clean(train_df)
    val_df = encode_and_clean(val_df)
    test_df = encode_and_clean(test_df)

    # 4. Build ClassLabel feature
    class_label = ClassLabel(names=ordered_labels)
    features = Features({
        text_column: Value("string"),
        label_column: class_label
    })

    # 5. Build HuggingFace Datasets with features
    train_dataset = Dataset.from_pandas(train_df, features=features)
    val_dataset = Dataset.from_pandas(val_df, features=features)
    test_dataset = Dataset.from_pandas(test_df, features=features)

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    

def downsample(df, downsampling_fraction):
    assert isinstance(downsampling_fraction, float)
    assert 0 <= downsampling_fraction < 1
    clean_df = df[df["label"] == "Clean"].sample(frac=downsampling_fraction, random_state=42)
    other_df = df[df["label"] != "Clean"]
    return pd.concat([clean_df, other_df]).reset_index(drop=True)


def main(args):
    data = read_data(args.data_path)
    df = create_dataframe(data)
    if args.downsample_clean < 1:
        print("Label distribution before downsampling Clean:")
        print(df["label"].value_counts())
        df = downsample(df, args.downsample_clean)
        print("Label distribution after downsampling Clean:")
        print(df["label"].value_counts())
    ds = stratified_split_and_encode(df, train_size=args.train_size)
    ds.save_to_disk(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script for preprocessing data in a trainable format."
    )

    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to raw data file."
    )
    parser.add_argument(
        "--save-path", type=str, required=True, help="Path to where the data will be saved."
    )
    parser.add_argument(
        "--train-size", type=float, default=0.7, help="Proportion of data reserved for train set, 0 < train-size < 1."
    )
    parser.add_argument(
        "--downsample-clean", type=float, default=0.2, help="Proportion of 'Clean' labels to keep. Set to 1 to keep all."
    )
    args = parser.parse_args()
    main(args)
