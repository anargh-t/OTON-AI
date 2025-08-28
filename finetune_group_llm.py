import argparse, json, os
import inspect
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import torch

def build_strategic_prompt(example):
    """Build prompt for strategic enterprise intelligence tasks"""
    instr = example.get("instruction","").strip()
    inp = example.get("input","").strip()
    resp = example.get("response","").strip()
    # Strategic prompt format for CXO-level insights
    prompt = f"### Strategic Task:\n{instr}\n\n### Business Context:\n{inp}\n\n### Strategic Analysis:\n{resp}"
    return prompt

def tokenize(example, tokenizer, max_len=1024):  # Increased max length for strategic content
    text = build_strategic_prompt(example)
    tokens = tokenizer(
        text, 
        truncation=True, 
        max_length=max_len,
        padding=False,
        return_tensors=None
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

class StrategicDataCollator:
    """Custom collator for strategic enterprise data"""
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        # Find the maximum length in this batch
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)
        
        # Pad all sequences to the same length
        batch = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            batch[key] = []
            for feature in features:
                if key in feature:
                    # Pad or truncate to max_len
                    if len(feature[key]) > max_len:
                        feature[key] = feature[key][:max_len]
                    else:
                        # Pad with appropriate token
                        if key == "labels":
                            padding_token = -100  # Ignore padding in loss calculation
                        else:
                            padding_token = self.tokenizer.pad_token_id
                        
                        feature[key] = feature[key] + [padding_token] * (max_len - len(feature[key]))
                    
                    batch[key].append(feature[key])
        
        # Convert to tensors
        for key in batch:
            batch[key] = torch.tensor(batch[key])
        
        return batch

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for Group-Level Strategic Intelligence")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--train_file", type=str, default="group_level_train.jsonl")
    parser.add_argument("--eval_file", type=str, default="group_level_eval.jsonl")
    parser.add_argument("--out_dir", type=str, default="group_level_outputs")
    parser.add_argument("--batch_size", type=int, default=2)  # Increased for TinyLlama
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)  # Lower learning rate for strategic tasks
    parser.add_argument("--max_len", type=int, default=1024)  # Longer sequences for strategic content
    parser.add_argument("--gradient_accumulation", type=int, default=2)  # Reduced for TinyLlama
    args = parser.parse_args()

    print(f"🚀 Starting Group-Level LLM Training")
    print(f"📊 Model: {args.model_name}")
    print(f"📁 Training data: {args.train_file}")
    print(f"📁 Evaluation data: {args.eval_file}")
    print(f"💾 Output directory: {args.out_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("🔄 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )

    # Apply LoRA - optimized for strategic enterprise intelligence
    print("🔧 Applying LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,  # Higher rank for complex strategic reasoning
        lora_alpha=64,  # Higher alpha for better adaptation
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()

    # Load datasets
    print("📚 Loading training and evaluation datasets...")
    train_ds = load_dataset("json", data_files={"train": args.train_file})["train"]
    eval_ds = load_dataset("json", data_files={"validation": args.eval_file})["validation"]

    print(f"📊 Training samples: {len(train_ds)}")
    print(f"📊 Evaluation samples: {len(eval_ds)}")

    # Tokenize datasets
    print("🔤 Tokenizing datasets...")
    train_ds = train_ds.map(lambda ex: tokenize(ex, tokenizer, args.max_len))
    eval_ds = eval_ds.map(lambda ex: tokenize(ex, tokenizer, args.max_len))

    # Create data collator
    collator = StrategicDataCollator(tokenizer=tokenizer, max_length=args.max_len)

    # Training arguments optimized for strategic tasks
    training_kwargs = dict(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=50,  # More frequent evaluation for strategic tasks
        logging_steps=25,
        save_strategy="steps",
        save_steps=100,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_accumulation_steps=args.gradient_accumulation,
        save_total_limit=3,
        bf16=False,
        dataloader_pin_memory=False,
        warmup_steps=100,  # Warmup for stable training
        lr_scheduler_type="cosine",  # Cosine learning rate schedule
        evaluation_strategy="steps",
        load_best_model_at_end=True,  # Load best model based on evaluation
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Filter out kwargs not supported by the installed transformers version
    supported_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in supported_params}
    training_args = TrainingArguments(**filtered_kwargs)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator
    )

    # Train the model
    print("🎯 Starting training...")
    trainer.train()
    
    # Save the model
    print("💾 Saving model and tokenizer...")
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    
    print("✅ Training completed successfully!")
    print(f"📁 Model saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
