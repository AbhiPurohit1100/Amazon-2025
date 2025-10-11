
# Step 2: Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

# Step 3: Load the CSV (train_fe.csv) into DataFrame
PATH = "train_fe.csv"

def load_robust_csv(path):
    import csv
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e1:
        print("C-engine failed:", e1)

    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", low_memory=False)
    except Exception as e2:
        print("python engine failed:", e2)

    # Try encodings
    for enc in ["utf-8-sig", "latin1"]:
        try:
            print(f"Trying encoding {enc} …")
            return pd.read_csv(path, engine="python", encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception as e3:
            print(f"Failed with {enc}:", e3)

    # Sniff delimiter (last resort)
    with open(path, "r", errors="replace") as f:
        sample = "".join([next(f) for _ in range(500)])
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","|","~","^","\t"])
        delim = dialect.delimiter
    print("Detected delimiter:", repr(delim))
    return pd.read_csv(path, engine="python", delimiter=delim, on_bad_lines="skip", low_memory=False)

# Load the CSV
df = load_robust_csv(PATH)
print(df.shape)
print(df.columns.tolist())
assert "catalog_content" in df.columns, "Expected 'catalog_content' column"

# Step 4: Set up device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "intfloat/e5-large-v2"

# Load tokenizer and pre-trained model for sequence classification
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)

# Step 5: Set up LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

# Apply LoRA adapters to the model
lora_model = get_peft_model(model, lora_config)
lora_model.train()

# Check trainable parameters
lora_model.print_trainable_parameters()

# Step 6: Prepare the dataset (FIXED)
class ProductDataset(Dataset):
    def __init__(self, texts, targets=None):
        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # FIXED: Remove return_tensors="pt" - let DataLoader handle batching
        inputs = tokenizer(text, padding="max_length", truncation=True, max_length=384)

        # Convert to tensors manually
        item = {key: torch.tensor(val) for key, val in inputs.items()}

        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item

# Assuming 'price' column exists for supervised training
y = np.log1p(df['price'].values).astype(np.float32)

# Create dataset and dataloader
train_dataset = ProductDataset(df["catalog_content"].tolist(), y)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Step 7: Optimizer and scheduler
optimizer = AdamW(lora_model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Step 8: Training loop (FIXED)
epochs = 3
for epoch in range(epochs):
    lora_model.train()
    epoch_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        optimizer.zero_grad()

        # FIXED: Move tensors to device properly
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Pass the inputs to the model
        outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask)

        # Get logits
        logits = outputs.logits.squeeze()

        # MSE loss for regression task
        loss = torch.nn.functional.mse_loss(logits, labels)

        # Backward pass & optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # FIXED: Scheduler step once per epoch, not per batch
    scheduler.step()
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(train_loader):.4f}")

# Step 9: Save the fine-tuned model
lora_model.save_pretrained("/content/fine_tuned_lora_model")
print("Fine-tuned model saved to /content/fine_tuned_lora_model")

# Step 10: Extract embeddings (FIXED)
def extract_embeddings(texts, batch_size=32):
    """
    Extract embeddings using the base model.
    AutoModelForSequenceClassification doesn't return hidden states by default,
    so we load the base AutoModel for embedding extraction.
    """
    # Load base model for embeddings
    base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    base_model.eval()

    embeddings_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True,
                             max_length=384, return_tensors="pt").to(device)
            outputs = base_model(**inputs)
            # Mean pooling over sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings_list.append(embeddings)

    return np.vstack(embeddings_list)

# Extract embeddings
print("Extracting embeddings...")
e_text_fine_tuned = extract_embeddings(df["catalog_content"].tolist())

# Step 11: Save the embeddings
np.save("/content/e_text_fine_tuned.npy", e_text_fine_tuned)
print(f"Embeddings saved to /content/train_e_text_fine_tuned.npy")
print(f"Embeddings shape: {e_text_fine_tuned.shape}")


PATH2 = "test_fe.csv"


# Load the CSV
df = load_robust_csv(PATH2)
print(df.shape)
print(df.columns.tolist())
assert "catalog_content" in df.columns, "Expected 'catalog_content' column"

print("Extracting embeddings...")
e_text_fine_tuned = extract_embeddings(df["catalog_content"].tolist())

# Step 11: Save the embeddings
np.save("/content/e_text_fine_tuned.npy", e_text_fine_tuned)
print(f"Embeddings saved to /content/test_e_text_fine_tuned.npy")
print(f"Embeddings shape: {e_text_fine_tuned.shape}")
