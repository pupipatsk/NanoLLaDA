import os
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
import uuid

# Modal image configuration with dependencies
stub = modal.Stub("llada-finetune")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pandas",
    "torch",
    "datasets",
    "tqdm",
    "transformers",
    "huggingface_hub",
)

# Configuration class
class Config:
    """Holds all configuration parameters for the script."""
    def __init__(self):
        self.repo_path = os.environ.get("REPO_PATH", "/app")
        self.data_dir = self.repo_path
        self.tokenized_data_dir = os.path.join(self.repo_path, "tokenized")
        self.model_name = "GSAI-ML/LLaDA-8B-Instruct"
        self.batch_size = int(os.environ.get("BATCH_SIZE", 2))
        self.lr = float(os.environ.get("LEARNING_RATE", 1e-5))
        self.num_epochs = int(os.environ.get("NUM_EPOCHS", 1))
        self.seed = int(os.environ.get("SEED", 42))
        self.mask_token_id = 126336
        self.sample_size = int(os.environ.get("SAMPLE_SIZE", 4000))
        self.device = self._select_device()

    def _select_device(self):
        """Selects the best available device (CUDA, MPS, or CPU)."""
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Using device: {device}")
        return device

# Data loading and preprocessing
def load_dataset_from_csv(train_file: str = None, sample_size: int = None) -> DatasetDict:
    """Loads dataset splits from CSV files and optionally samples rows."""
    split_files = {"train": train_file}
    dct = {}
    for split, file_path in split_files.items():
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if sample_size:
                df = df.sample(sample_size, random_state=Config().seed)
            dct[split] = Dataset.from_pandas(df)
    return DatasetDict(dct)

def format_llada_prompt(example, tokenizer):
    """Formats an example into a prompt for the LLaDA model and tokenizes it."""
    instruction = f"<start_id>user<end_id>\nสรุปข้อความต่อไปนี้\n{example['body']}<eot_id><start_id>assistant<end_id>\n{example['summary']}<EOS>"
    tokenized = tokenizer(instruction, padding="max_length", truncation=True, max_length=2048)
    prompt_end = instruction.find("<start_id>assistant<end_id>")
    prompt_tokens = tokenizer(instruction[:prompt_end])["input_ids"]
    return {"input_ids": tokenized["input_ids"], "prompt_length": len(prompt_tokens)}

def load_and_preprocess_data(config: Config, tokenizer) -> tuple:
    """Loads and preprocesses the dataset, saving tokenized data to disk."""
    train_data_file = os.path.join(config.data_dir, "train-7000-1024.csv")
    if not os.path.exists(train_data_file):
        raise FileNotFoundError(f"Training data file not found: {train_data_file}")
    dataset_dict = load_dataset_from_csv(train_file=train_data_file, sample_size=config.sample_size)
    os.makedirs(config.tokenized_data_dir, exist_ok=True)
    processed_data = dataset_dict["train"].map(lambda x: format_llada_prompt(x, tokenizer))
    output_path = os.path.join(config.tokenized_data_dir, "train.jsonl")
    processed_data.to_json(output_path)
    print(f"Saved tokenized data to: {output_path}")
    return processed_data

# Model loading
def load_model(config: Config) -> AutoModel:
    """Loads the LLaDA model and prepares it for training."""
    print(f"Loading {config.model_name} model...")
    model = AutoModel.from_pretrained(config.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(config.device)
    model.train()
    print(f"{config.model_name} model loaded successfully.")
    return model

# DataLoader collate function
def collate_fn(batch):
    """Prepares a batch for training by converting to tensors."""
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    prompt_lengths = torch.tensor([item["prompt_length"] for item in batch])
    return {"input_ids": input_ids, "prompt_lengths": prompt_lengths}

# Training function
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

# Generation functions
def add_gumbel_noise(logits, temperature):
    """Adds Gumbel noise to logits for sampling."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    """Calculates the number of tokens to transfer at each generation step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """Generates text using the trained model."""
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        start_idx = prompt.shape[1] + num_block * block_length
        end_idx = start_idx + block_length
        block_mask_index = (x[:, start_idx:end_idx] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            if cfg_scale > 0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand_like(x0, device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented.")

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

# Modal job entry point
@stub.function(
    image=image,
    gpu="A100",  # Specify GPU type (adjust based on needs)
    timeout=7200,  # 2 hours timeout
    secrets=[modal.Secret.from_name("huggingface-token")],  # Hugging Face token secret
)
def run_finetune():
    config = Config()
    torch.manual_seed(config.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    # Load and preprocess data
    processed_data = load_and_preprocess_data(config, tokenizer)

    # Load model
    model = load_model(config)

    # Prepare DataLoader
    dataloader = DataLoader(
        processed_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Train model
    optimizer = AdamW(model.parameters(), lr=config.lr)
    train_model(
        model, dataloader, optimizer, config.num_epochs, config.device, config.mask_token_id
    )

    # Inference
    model.eval()
    prompt_text = (
        "สรุปข้อความต่อไปนี้\n ความเก่ง เกิดขึ้นได้หลายแบบไม่ว่าจะ "
        "ความหมั่นเพียร(ฝึกซ้อม), ประสบการณ์, สิ่งแวดล้อมเกื้อหนุน, มีต้นทุนบางอย่างดี "
        "เหมือนคนเกิดมาร่างกายสูงใหญ่มีโอกาสเก่งในกีฬาหลายประเภท นี่ก็ถือว่าต้นทุนดี "
        "แต่เหล่านี้เองจึงย้อนไปบั่นทอนคนที่คิดว่าตนไม่เก่ง เช่น เราขี้เกียจ-ไม่มีเวลาซ้อม, "
        "เราไม่เคยทำมาก่อน, ยังไม่พร้อม, ต้นทุนไม่ดีเหมือนเขา ส่วนหนึ่งก็ใช่ว่าผิด "
        "แต่แน่นอนไม่ถูก และกลายเป็นถ่วงอนาคตอย่างมาก"
    )
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = torch.tensor(tokenizer(prompt)["input_ids"]).to(config.device).unsqueeze(0)
    output_ids = generate(
        model,
        input_ids,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=config.mask_token_id,
    )
    summary = tokenizer.batch_decode(
        output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
    )[0]
    print("Generated Summary:")
    print(summary)

    # Push model to Hugging Face Hub
    from huggingface_hub import login
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
        model.push_to_hub("pupipatsk/llada-thaisum-finetuned")
        tokenizer.push_to_hub("pupipatsk/llada-thaisum-finetuned")
        print("Model and tokenizer pushed to Hugging Face Hub.")
    else:
        print("HUGGINGFACE_TOKEN not found. Skipping model push.")

if __name__ == "__main__":
    with stub.run():
        run_finetune.call()