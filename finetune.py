# Step 1: Import required libraries
import os
import torch
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup  # For HTML parsing
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Step 2: Configure GPU Memory Optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Step 3: Define Model and Tokenizer
model_name = "deepseek-ai/deepseek-coder-5.7bmqa-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define 4-bit quantization config with optimizations
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model with quantization config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

# Step 4: Attach LoRA Adapters
lora_config = LoraConfig(
    r=2,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Step 5: Function to extract text from files
def extract_text_from_file(file_path):
    """Extract text from a file based on its extension."""
    try:
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        elif file_path.endswith(('.txt', '.md')):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_path.endswith('.html'):
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                return soup.get_text(separator='\n')  # Extract text, separate elements with newlines
        else:
            return ""  # Skip unsupported file types
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# Step 6: Process all files in the directory and subdirectories
input_dir = "C:/Users/kalaivani/Desktop/Genai/blender_python_reference_4_3"
output_text_file = "C:/Users/kalaivani/Desktop/Genai/animation/traindata/preprocessed_blender_docs.txt"

os.makedirs(os.path.dirname(output_text_file), exist_ok=True)

all_text = ""
for root, _, files in os.walk(input_dir):
    for file in files:
        file_path = os.path.join(root, file)
        text = extract_text_from_file(file_path)
        if text:
            all_text += text + "\n"
            print(f"Processed: {file_path}")  # Optional: Log processed files

# Save the combined text to a single file
with open(output_text_file, 'w', encoding='utf-8') as f:
    f.write(all_text)
print(f"Preprocessed text saved to {output_text_file}")

# Step 7: Load the preprocessed dataset
dataset = load_dataset("text", data_files={"train": output_text_file})

# Step 8: Sliding Window Tokenization (reduced window size)
def sliding_window_tokenize(examples, window_size=128, stride=64):
    tokenized_inputs = tokenizer(
        examples["text"], truncation=False, padding=False, add_special_tokens=True
    )
    input_ids = tokenized_inputs["input_ids"]

    chunked_inputs = {
        "input_ids": [],
        "attention_mask": []
    }

    for ids in input_ids:
        for i in range(0, len(ids), stride):
            chunk = ids[i:i + window_size]
            if len(chunk) < window_size:
                chunk += [tokenizer.pad_token_id] * (window_size - len(chunk))
            chunked_inputs["input_ids"].append(chunk)
            chunked_inputs["attention_mask"].append([1] * len(chunk))

    return chunked_inputs

tokenized_datasets = dataset.map(sliding_window_tokenize, batched=True, remove_columns=["text"])

# Step 9: Create Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 10: Define Training Arguments (optimized for low VRAM)
output_dir = "C:/Users/kalaivani/Desktop/Genai/animation/op"

last_checkpoint = None
if boneless.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
    last_checkpoint = max(
        [os.path.join(output_dir, d) for d in os.listdir(output_dir) if "checkpoint" in d],
        key=os.path.getmtime,
        default=None
    )

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,  # Increase if needed
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=1000,
    save_strategy="steps",
    save_total_limit=2,
    logging_steps=200,
    fp16=True,
    remove_unused_columns=False,
    report_to="none",
    load_best_model_at_end=False,
    resume_from_checkpoint=last_checkpoint if last_checkpoint else None,
    eval_strategy="no"
)

# Step 11: Create Trainer Instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"]
)

# Step 12: Clear GPU Cache Before Training
torch.cuda.empty_cache()

# Step 13: Start Training
if last_checkpoint:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("No checkpoint found. Starting fresh training...")
    trainer.train()

# Step 14: Save Fine-Tuned Model
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")