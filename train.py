import torch
from huggingface_hub import login
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import setup_chat_format, SFTTrainer
from peft import LoraConfig
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

login(
	token="###",
	add_to_git_credential=False
)

dataset1 = load_dataset("cais/mmlu", name="high_school_us_history", split="test[:100]")
dataset2 = load_dataset("cais/mmlu", name="elementary_mathematics", split="test[:100]")
dataset3 = load_dataset("cais/mmlu", name="high_school_computer_science", split="test[:100]")

def format_mmlu(example):
    choices = example["choices"]
    choices_text = "\n".join(
        [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
    )
    
    prompt = (
        f"You are a knowledgeable AI assistant.\n"
        f"Read the following multiple choice question carefully and choose the correct answer.\n\n"
        f"Question: {example['question']}\n\n"
        f"{choices_text}\n\n"
        f"Answer with the letter corresponding to the correct option (A, B, C, or D):"
    )
    
    return {"text": prompt}

dataset1 = dataset1.map(format_mmlu, remove_columns=dataset1.column_names)
dataset2 = dataset2.map(format_mmlu, remove_columns=dataset2.column_names)
dataset3 = dataset3.map(format_mmlu, remove_columns=dataset3.column_names)

dataset = interleave_datasets([dataset1, dataset2, dataset3])
model_id = "###"

bnb_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_use_double_quant=True,
	bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
	model_id,
	device_map={'':torch.cuda.current_device()},
	torch_dtype=torch.bfloat16,
	quantization_config=bnb_config,
	use_safetensors=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_size = 'right'

peft_config = LoraConfig(
	lora_alpha=128,
	lora_dropout=0.05,
	r=256,
	bias="none",
	target_modules=[
		# ...
	],
	task_type="CAUSAL_LM",
)

args = TrainingArguments(
	output_dir="code-llama-7b-text-to-sql",
	num_train_epochs=3,
	per_device_train_batch_size=1,
	gradient_accumulation_steps=4,
	gradient_checkpointing=True,
	gradient_checkpointing_kwargs={"use_reentrant": False},
	optim="adamw_torch_fused",
	logging_steps=10,
	save_strategy="epoch",
	learning_rate=2e-4,
	bf16=True,
	tf32=False,
	max_grad_norm=0.3,
	warmup_ratio=0.03,
	lr_scheduler_type="constant",
	push_to_hub=True,
	report_to="tensorboard"
)

def preprocess(example):
    tokenized = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(preprocess)

max_seq_length = 512

trainer = SFTTrainer(
	model=model,
	args=args,
	train_dataset=dataset,
	peft_config=peft_config,
)

trainer.train()
trainer.save_model()
