# ==========================================
# 2. MODEL, TOKENIZER, AND PPO SETUP
# ==========================================
# Using a small model for demonstration. Replace with your SFT base model.

import torch
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig

# Import your custom referee from your existing file
from judge import z3_referee_reward
model_id = "Qwen/Qwen1.5-0.5B" 

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA Configuration to prevent OOM errors
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

print("Loading model with Value Head and LoRA...")
# PPOTrainer requires a Value Head to estimate the expected reward (Critic)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_id,
    peft_config=lora_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
# Create a frozen reference model to calculate KL divergence
ref_model = create_reference_model(model)

ppo_config = PPOConfig(
    model_name=model_id,
    learning_rate=1.41e-5,
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=2,
    optimize_cuda_cache=True,
    seed=42
)

ppo_trainer = PPOTrainer(
    config=ppo_config, 
    model=model, 
    ref_model=ref_model, 
    tokenizer=tokenizer
)

# ==========================================
# 3. DUMMY DATASET PREPARATION
# ==========================================
# In production, load this from your prompt database
raw_prompts = [
    "Schedule a senior engineer for DB migration this Saturday under 2000.",
    "I need someone for basic maintenance on a weekday, cost doesn't matter.",
    "Can a junior handle the weekend deployment?",
    "Book a mid-level dev for Saturday maintenance, keep it cheap."
]

# Format prompts into tensors
prompt_tensors = [tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(ppo_trainer.accelerator.device) for prompt in raw_prompts]

# ==========================================
# 4. THE PPO TRAINING LOOP
# ==========================================
epochs = 3
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 64, # Enough to generate the JSON
}

print("Starting PPO Training...")

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch+1}/{epochs} ---")
    
    # Process in batches 
    for i in range(0, len(prompt_tensors), ppo_config.batch_size):
        batch_queries = prompt_tensors[i:i + ppo_config.batch_size]
        
        # 1. LLM generates the JSON responses
        batch_responses = []
        for query in batch_queries:
            # Generate response tensor
            response = ppo_trainer.generate(query, **generation_kwargs)
            # Remove the prompt from the generated tensor to isolate the response
            response_only = response.squeeze()[-generation_kwargs["max_new_tokens"]:]
            batch_responses.append(response_only)
        
        # Decode to text for the Z3 referee
        response_texts = [tokenizer.decode(r, skip_special_tokens=True) for r in batch_responses]
        
        # 2. Score the batch using the Z3 Referee
        rewards = []
        for response_text in response_texts:
            score = z3_referee_reward(response_text)
            # TRL expects rewards as list of float tensors
            rewards.append(torch.tensor(score, dtype=torch.float32).to(ppo_trainer.accelerator.device))
            
        print(f"Generated examples:\n{response_texts[0]}\nReward: {rewards[0].item()}")

        # 3. PPO Step: Update actor and critic models
        # This calculates advantages, KL penalty, and updates weights
        stats = ppo_trainer.step(batch_queries, batch_responses, rewards)
        
        print(f"Batch Mean Reward: {torch.stack(rewards).mean().item():.4f} | Objective/KL: {stats['objective/kl']:.4f}")

print("\nTraining Complete. Saving model...")
ppo_trainer.save_pretrained("./neuro_symbolic_agent")
tokenizer.save_pretrained("./neuro_symbolic_agent")
