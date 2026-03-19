import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# 1. Load your base model (e.g., Llama-3-8B)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# 2. Load your Golden Trajectories dataset
dataset = load_dataset("json", data_files="train.jsonl", split="train")

# 3. Format the dataset into the model's expected chat template
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['messages'])):
        text = tokenizer.apply_chat_template(example['messages'][i], tokenize=False)
        output_texts.append(text)
    return output_texts

# 4. Mask the user prompts so the model ONLY learns from the JSON generation
# This ensures the model learns to generate the answer, not predict the prompt
response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# 5. Configure and run the SFT Trainer
training_args = TrainingArguments(
    output_dir="./sft_bootstrapped_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=10,
    max_seq_length=1024,
    num_train_epochs=3,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
