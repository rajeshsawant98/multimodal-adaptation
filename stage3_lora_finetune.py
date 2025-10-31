# stage3_lora_text_proxy.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------
# Load mini datasets
# --------------------
def load_dataset(path):
    df = pd.read_parquet(path)
    df = df.dropna(subset=['caption'])
    # Create dummy "image features" column as proxy
    df['image_features'] = ["dummy features"] * len(df)
    return df

coco_df = load_dataset("data/cache/coco_mini.parquet")
flickr_df = load_dataset("data/cache/flickr_mini.parquet")
data_df = pd.concat([coco_df, flickr_df]).reset_index(drop=True)

# --------------------
# Load tokenizer + T5
# --------------------
base_model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)
t5_model.train()

# --------------------
# LoRA adapter
# --------------------
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
)
t5_model = get_peft_model(t5_model, lora_config)
print("✅ LoRA adapter attached")

# --------------------
# Preprocess for LoRA
# --------------------
def preprocess(example):
    # Use text proxy for "image features"
    input_text = f"Image features: {example['image_features']}"
    target_text = example['caption']

    tokenized_inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=128)
    tokenized_labels = tokenizer(target_text, truncation=True, padding="max_length", max_length=64)
    tokenized_inputs["labels"] = tokenized_labels["input_ids"]
    return tokenized_inputs

hf_dataset = Dataset.from_pandas(data_df)
tokenized_dataset = hf_dataset.map(preprocess, remove_columns=hf_dataset.column_names)
print("✅ Dataset tokenized")

# --------------------
# Training args
# --------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="lora_output",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=1,
    save_steps=2,
    save_total_limit=1,
    learning_rate=3e-4,
    fp16=torch.cuda.is_available(),
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=t5_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# --------------------
# Train LoRA
# --------------------
trainer.train()
t5_model.save_pretrained("lora_adapter")
print("✅ Stage 3 LoRA fine-tuning complete. Adapter saved.")
