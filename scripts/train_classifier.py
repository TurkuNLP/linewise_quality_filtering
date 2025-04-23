import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for consistency

import comet_ml #type:ignore
import argparse
from pathlib import Path
import numpy as np
from datasets import load_from_disk,  load_dataset #type:ignore
from sklearn.metrics import ( #type:ignore
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight #type:ignore
import torch #type:ignore
from torch.nn import CrossEntropyLoss #type:ignore
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

class TrainerWithWeightsAndSmoothing(Trainer):
    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if torch.isnan(logits).any():
            print("WARNING: NaNs in logits before loss")
  
        # Send class_weights to correct device
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        else:
            weight = None

        # Loss function with class weights and label smoothing
        loss_fct = CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# Pass label_encoder to use class names in the classification report
def compute_metrics(pred, label_names):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    if experiment:
        epoch = int(experiment.curr_epoch) if experiment.curr_epoch is not None else 0
        experiment.set_epoch(epoch)
        experiment.log_confusion_matrix(
            y_true=labels,
            y_predicted=preds,
            file_name=f"confusion-matrix-epoch-{epoch}.json",
            labels=label_names,
        )

    # Accuracy
    accuracy = accuracy_score(labels, preds)

    # F1 Score
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

    # Precision and Recall
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)

    # Generate the classification report using class names
    class_report = classification_report(labels, preds, target_names=label_names, zero_division=0)

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
    }

def optimization_config():
    return {
    "algorithm": "bayes",
    "parameters": {
        "learning_rate": {"type": "float", "scaling_type": "log_uniform", "min": 0.00001, "max": 0.001},
        "batch_size": {"type": "discrete", "values": [32, 64, 128]},
    },

    # Declare what to optimize, and how:
    "spec": {
      "maxCombo": 20,
      "metric": "loss",
      "objective": "minimize",
    },
    }

def calculate_class_weights(dataset):
    labels = dataset["train"]["label"]
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_labels,
        y=labels,
    )
    return torch.tensor(class_weights, dtype=torch.float)

# Main function to run the training process
def main(args):
    
    if args.optimize:
        opt = comet_ml.Optimizer(config=optimization_config())

    else:
        global experiment
        experiment = comet_ml.start(
            api_key = os.environ["COMET_API_KEY"],
            project_name="linewise-quality-filtering"
        )
        os.environ["COMET_LOG_ASSETS"] = "True"

    # Load data
    # dataset = load_from_disk(args.data_path)
    dataset = load_dataset('HPLT/HPLT2.0_cleaned', data_files='eng_Latn_1/train-00000-of-00356.parquet')
    
    
    # Get model path
    saved_model_path = Path(".") / "results" / "finetuned_models" / str(args.run_id)
    saved_model_path.mkdir(parents=True, exist_ok=True)

    # If training, load the base model; if not, load the saved model
    load_model_name = args.base_model
    
    print(f"Finetuning model: {load_model_name}")

    # Get label names and num_labels
    label_names = dataset["train"].features["label"].names
    num_labels = len(label_names)
    
    label2id = {label: id for id, label in enumerate(label_names)}
    id2label = {id: label for id, label in enumerate(label_names)}
    
    print(f"Label2id: {label2id}")
    print(f"Id2label: {id2label}")
    print("Unique labels in train set:", np.unique(dataset["train"]["label"]))
    print("Number of labels:", num_labels)

    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="longest",
            truncation=True,
            max_length=512,
        )

    dataset = dataset.map(tokenize, batched=True)
    
    print(f"Example tokenized input: {dataset['train'][0]}") 
    
    #Calculate class weigths because label distribution is not equal
    class_weights = calculate_class_weights(dataset)
    
    print("Class weights:", class_weights)

    # Shuffle the train split
    dataset["train"] = dataset["train"].shuffle(seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        load_model_name,
        num_labels=num_labels,
        trust_remote_code=True,
        label2id=label2id,
        id2label=id2label,
    )
    
    if args.test or args.train:
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=saved_model_path,
            learning_rate=args.learning_rate,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            logging_dir=saved_model_path / "logs",
            logging_steps=100,
            save_steps=200,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,
            num_train_epochs=4,
            seed=42,
            fp16=True,
            max_grad_norm=1.0, 
            group_by_length=True,
            report_to=["comet_ml"],
            disable_tqdm=True,
        )

        trainer = TrainerWithWeightsAndSmoothing(
            model=model,
            args=training_args,
            train_dataset=dataset["train"].shard(num_shards=5, index=0),
            eval_dataset=dataset["validation"].shard(num_shards=5, index=0),
            processing_class=tokenizer,
            compute_metrics=lambda pred: compute_metrics(pred, label_names),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            class_weights=class_weights,
            label_smoothing=0.1,
        )
        
        if args.train:
            # Train the model
            trainer.train()
            trainer.save_model(saved_model_path)

        # Evaluate the model on the test set
        eval_result = trainer.evaluate(eval_dataset=dataset["test"])
        with open(saved_model_path / "eval_results.txt", "w") as f:
            f.write(f"Test set evaluation results: \n")
            for key, value in eval_result.items():
                f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="FacebookAI/xlm-roberta-large" )
    parser.add_argument("--learning-rate", type=float, default=0.00001)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    main(args)

