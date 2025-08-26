import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin
from tqdm import tqdm
import pandas as pd
import json
from pathlib import Path

# Your model definition remains the same
class QualityModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(QualityModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

config = AutoConfig.from_pretrained("nvidia/quality-classifier-deberta")
tokenizer = AutoTokenizer.from_pretrained("nvidia/quality-classifier-deberta")
model = QualityModel.from_pretrained("nvidia/quality-classifier-deberta").to(device)

model.eval()

texts = []
quality_scores = []

data_path = Path("../data/hplt/eng_Latn_small_sample/1_00.jsonl")
with data_path.open("r") as data:
    for i in data:
        d = json.loads(i)
        texts.append(d["text"])
        quality_scores.append(d["doc_scores"][0])
        if len(texts) == 100_000:
            break
        
def predict(batch):
    with torch.no_grad():
        inputs = tokenizer(
            batch, return_tensors="pt", padding="longest", truncation=True
            ).to(device)
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        predicted_classes = torch.argmax(outputs, dim=1)
        predicted_domains = [
            config.id2label[class_idx.item()] for class_idx in predicted_classes.cpu().numpy()
        ]
        
        return predicted_domains

batch = []
preds = []
for text in tqdm(texts):
    batch.append(text)
    if len(batch) == 64:  # Process in batches of 64
        preds.extend(predict(batch))
        batch = []

if batch:
    preds.extend(predict(batch))

# Convert predictions to DataFrame
df = pd.DataFrame({
    "text": texts[:len(preds)],
    "nemotron_label": preds,
    "hplt_doc_score": quality_scores[:len(preds)]
})

# Save DataFrame to CSV
df.to_csv("../data/nemotron_quality_predictions.csv", index=False)