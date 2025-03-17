import os

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for consistency

import argparse

import numpy as np
import platt_scaler
import predict_fineweb
from datasets import DatasetDict, load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


class CustomTrainer(Trainer):
    def __init__(self, *args, label_smoothing, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.loss_fct = CrossEntropyLoss(label_smoothing=label_smoothing)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels from inputs
        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Compute loss with label smoothing
        loss = self.loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# Pass label_encoder to use class names in the classification report
def compute_metrics(pred, label_encoder):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    # Calculate confusion matrix (optional)
    conf_matrix = confusion_matrix(labels, preds).tolist()

    # Accuracy
    accuracy = accuracy_score(labels, preds)

    # F1 Score
    f1 = f1_score(labels, preds, average="weighted")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")

    # Precision and Recall
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")

    # Use label_encoder to get class names
    class_names = label_encoder.classes_

    # Generate the classification report using class names
    class_report = classification_report(labels, preds, target_names=class_names)

    # Print the classification report
    print("Classification Report:\n", class_report)

    # Return metrics
    return {
        "accuracy": accuracy,
        "f1": f1,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix,
    }


# Main function to run the training process
def main(args):

    # Load each split into a Dataset
    data_files = {
        "train": f"data/train.jsonl",
        "test": f"data/test.jsonl",
        "dev": f"data/dev.jsonl",
    }

    # Load the JSONL files as a DatasetDict
    dataset = DatasetDict(
        {
            split_name: load_dataset(
                "json",
                data_files={split_name: file_path},
                split=split_name,
            )
            for split_name, file_path in data_files.items()
        }
    )

    # Initialize and fit LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset["train"]["label"])

    # Function to encode labels in a dataset
    def encode_labels(example):
        example["label"] = label_encoder.transform([example["label"]])[0]
        return example

    # Apply the transformation to each split
    dataset = dataset.map(encode_labels)

    # Get model path
    saved_model_name = args.finetuned_model_path or (
        f"{args.labels}_finetuned_"
        + args.base_model.replace("/", "_")
        + ("_with_synth" if args.add_synthetic_data else "")
    )

    # If training, load the base model; if not, load the saved model
    load_model_name = args.base_model if args.train else saved_model_name

    num_labels = len(label_encoder.classes_)

    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(
            batch["line" if args.labels == "llm" else "text"],
            padding="longest",
            truncation=True,
            max_length=512,
        )

    dataset = dataset.map(tokenize, batched=True)

    # Shuffle the train split
    dataset["train"] = dataset["train"].shuffle(seed=42)

    print("Example of a tokenized input:")
    print(dataset["train"][0])

    model = AutoModelForSequenceClassification.from_pretrained(
        load_model_name, num_labels=num_labels
    )

    if args.test or args.train:
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=saved_model_name,
            learning_rate=args.learning_rate or 0.00001,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            logging_dir=f"{saved_model_name}/logs",
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            num_train_epochs=5,
            seed=42,
            bf16=True,
            tf32=True,
            group_by_length=True,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["dev"],
            tokenizer=tokenizer,
            compute_metrics=lambda pred: compute_metrics(pred, label_encoder),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            label_smoothing=0.1,
        )
        if args.train:
            # Train the model
            trainer.train()
            trainer.save_model(saved_model_name)

        # Evaluate the model on the test set
        eval_result = trainer.evaluate(eval_dataset=dataset["test"])
        print(f"Test set evaluation results: {eval_result}")

    if args.platt or args.platt_tune:
        platt_scaler.run(
            saved_model_name,
            model.to("cuda"),
            tokenizer,
            dataset["test"],
            label_encoder,
            args.platt_tune,
        )

    if args.predict_fineweb:
        predict_fineweb.run(
            saved_model_name, model.to("cuda"), tokenizer, label_encoder, "clean"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--labels", type=str, default="free")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--pooling_type", type=str, default="cls")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--finetuned_model_path", type=str)
    parser.add_argument("--platt", action="store_true")
    parser.add_argument("--platt_tune", action="store_true")
    parser.add_argument("--predict_fineweb", action="store_true")
    args = parser.parse_args()

    main(args)
    