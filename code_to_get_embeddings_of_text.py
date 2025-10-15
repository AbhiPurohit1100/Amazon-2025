import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# === Auto directory setup ===
BASE_DIR = os.getcwd()
print(f"📁 Using BASE_DIR: {BASE_DIR}")

TRAIN_PATH = os.path.join(BASE_DIR, "train_fe.csv")
TEST_PATH  = os.path.join(BASE_DIR, "test_fe.csv")

# === Choose which checkpoint to load ===
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "fine_tuned_epoch1")  # change if needed
MODEL_NAME = "intfloat/e5-large-v2"

# === Device setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Using device: {device}")

# === Load tokenizer and base model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# === Load LoRA fine-tuned weights ===
if os.path.exists(CHECKPOINT_DIR):
    print(f"🔄 Loading fine-tuned LoRA checkpoint from {CHECKPOINT_DIR}")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
else:
    print("⚠️ Checkpoint not found, using base model instead.")
    model = base_model

model.eval()

# === Function to extract embeddings ===
def extract_embeddings(df, text_col="catalog_content", batch_size=32):
    texts = df[text_col].astype(str).tolist()
    embeddings_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {text_col} embeddings"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            # Mean pool over sequence length
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings_list.append(emb)

    return np.vstack(embeddings_list)

# === Process TRAIN ===
print("\n📦 Extracting train embeddings...")
train_df = pd.read_csv(TRAIN_PATH)
train_emb = extract_embeddings(train_df)

np.save(os.path.join(BASE_DIR, "train_embeddings.npy"), train_emb)
print(f"✅ Saved train embeddings to {os.path.join(BASE_DIR, 'train_embeddings.npy')}")
print(f"🔢 Shape: {train_emb.shape}")

# Optional: Add to DataFrame for analysis
train_df["embeddings"] = train_emb.tolist()
train_df.to_pickle(os.path.join(BASE_DIR, "train_with_embeddings.pkl"))
print("💾 Saved train_with_embeddings.pkl")

# === Process TEST ===
print("\n📦 Extracting test embeddings...")
test_df = pd.read_csv(TEST_PATH)
test_emb = extract_embeddings(test_df)

np.save(os.path.join(BASE_DIR, "test_embeddings.npy"), test_emb)
print(f"✅ Saved test embeddings to {os.path.join(BASE_DIR, 'test_embeddings.npy')}")
print(f"🔢 Shape: {test_emb.shape}")

test_df["embeddings"] = test_emb.tolist()
test_df.to_pickle(os.path.join(BASE_DIR, "test_with_embeddings.pkl"))
print("💾 Saved test_with_embeddings.pkl")

print("\n🎯 All embeddings extracted successfully!")
