# -- Imports --
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import numpy as np
import modal 

app = modal.App("fine-tune-llada")

weight_volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

model_id = "GSAI-ML/LLaDA-8B-Instruct"
model_path = os.path.join(MODEL_DIR, model_id)


data_volume = modal.Volume.from_name("data-vol", create_if_missing=True)
DATA_DIR = Path("/data")

tokenized_data_dir = os.path.join(DATA_DIR, "tokenized")


batch_size = 2
num_epochs = 1
lr = 1e-5
mask_token_id = 126336

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pandas",
    "torch",
    "datasets",
    "tqdm",
    "transformers",
    "huggingface_hub",
)

def select_device():
    """Selects the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device


def load_dataset_from_csv(train_file: str = None, valid_file: str = None, test_file: str = None, sample_size: int = None) -> DatasetDict:
    """Loads dataset splits from CSV files and optionally samples rows."""
    split_files = {"train": train_file, "validation": valid_file, "test": test_file}
    dct = {}
    for split, file_path in split_files.items():
        if file_path:
            df = pd.read_csv(file_path)
            if sample_size:
                df = df.sample(sample_size)
            dct[split] = Dataset.from_pandas(df)
    return DatasetDict(dct)

def format_llada_prompt(example, tokenizer):
    """Formats an example into a prompt for the LLaDA model and tokenizes it."""
    instruction = f"<start_id>user<end_id>\nสรุปข้อความต่อไปนี้\n{example['body']}<eot_id><start_id>assistant<end_id>\n{example['summary']}<EOS>"
    tokenized = tokenizer(instruction, padding="max_length", truncation=True, max_length=2048)
    prompt_end = instruction.find("<start_id>assistant<end_id>")
    prompt_tokens = tokenizer(instruction[:prompt_end])["input_ids"]
    return {"input_ids": tokenized["input_ids"], "prompt_length": len(prompt_tokens)}

def load_and_preprocess_data(sample_size: int = 100) -> tuple:
    """Loads and preprocesses the dataset, saving tokenized data to disk."""
    train_data_file = os.path.join(DATA_DIR, "train.csv")
    dataset_dict = load_dataset_from_csv(train_file=train_data_file, sample_size=sample_size)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        # torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    os.makedirs(tokenized_data_dir, exist_ok=True)
    processed_data = dataset_dict["train"].map(lambda x: format_llada_prompt(x, tokenizer))
    output_path = os.path.join(tokenized_data_dir, "train.jsonl")
    processed_data.to_json(output_path)
    print(f"Saved tokenized data to: {output_path}")
    return processed_data, tokenizer



# DataLoader Collate Function
def collate_fn(batch):
    """Prepares a batch for training by converting to tensors."""
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    prompt_lengths = torch.tensor([item["prompt_length"] for item in batch])
    return {"input_ids": input_ids, "prompt_lengths": prompt_lengths}

# Training Function
def train_model(model: AutoModel, dataloader: DataLoader, optimizer: AdamW, num_epochs: int, device: str, mask_token_id: int):
    """Trains the model using a masked language modeling approach."""
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)

            # Create noisy batch by masking post-prompt tokens
            noisy_batch = input_ids.clone()
            for i in range(noisy_batch.shape[0]):
                noisy_batch[i, prompt_lengths[i]:] = mask_token_id
            mask_index = (noisy_batch == mask_token_id)

            # Forward pass
            logits = model(input_ids=noisy_batch).logits
            p_mask = torch.ones_like(noisy_batch, dtype=torch.float32).to(device)

            # Compute loss only on masked tokens
            token_loss = F.cross_entropy(logits[mask_index], input_ids[mask_index], reduction="none") / p_mask[mask_index]
            loss = token_loss.sum() / input_ids.shape[0]

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item())


@app.function(volumes={MODEL_DIR: weight_volume, DATA_DIR: data_volume}, image=image, gpu="A100-80GB", timeout=18000)
def run_finetune():
    # Load and preprocess data
    processed_data, tokenizer = load_and_preprocess_data(config, sample_size=4000)
    
    # Model Loading
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True  # Ensure it only loads from local volume
    )
    model.to(select_device())
    model.train()
    
    
    # Prepare DataLoader
    dataloader = DataLoader(
        processed_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    
    # Train model
    optimizer = AdamW(model.parameters(), lr=lr)
    train_model(
        model, dataloader, optimizer, num_epochs, select_device(), mask_token_id
    )
    
    print("Training complete. Saving model...")
    from huggingface_hub import login
    login(token="")
    model.push_to_hub("pupipatsk/llada-thaisum-finetuned")