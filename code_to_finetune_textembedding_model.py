import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig

# === Paths ===
BASE_DIR = os.getcwd()
print(f"📁 Using BASE_DIR: {BASE_DIR}")

TRAIN_PATH = os.path.join(BASE_DIR, "train_fe.csv")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load data ===
df = pd.read_csv(TRAIN_PATH)
y = np.log1p(df["price"].values).astype(np.float32)

MODEL_NAME = "intfloat/e5-large-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Training on device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)

# === Apply LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)
lora_model = get_peft_model(model, lora_config)
lora_model.train()

# === Dataset ===
class ProductDataset(Dataset):
    def __init__(self, texts, targets=None):
        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=384
        )
        item = {k: torch.tensor(v) for k, v in inputs.items()}
        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item

train_dataset = ProductDataset(df["catalog_content"].tolist(), y)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# === Optimizer ===
optimizer = AdamW(lora_model.parameters(), lr=1e-4)

# === Training with checkpoint saving ===
EPOCHS = 3
for epoch in range(EPOCHS):
    lora_model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()
        loss = torch.nn.functional.mse_loss(logits, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

    # --- Save model checkpoint ---
    epoch_dir = os.path.join(SAVE_DIR, f"fine_tuned_epoch{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    lora_model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)
    print(f"💾 Model saved to {epoch_dir}")

print("🎯 Training complete! All checkpoints saved.")
