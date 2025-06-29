import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
from datasets import load_dataset
import re

def load_model_local(base_model_id, adapter_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map={'': torch.cuda.current_device()},
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        use_safetensors=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.padding_side = 'right'
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer

def clean(text):
    text = re.sub(r"<\|.*?\|>", "", text)
    return text.strip().split("\n")[0].lower().strip()

def extract_letter(text):
    match = re.search(r"\b([A-Da-d])\b", text)
    return match.group(1).lower() if match else ""

def evaluate_on_selected_mmlu(model, tokenizer, max_samples_per_subject=50):
    subjects = [
        "high_school_us_history",
        "elementary_mathematics",
        "high_school_computer_science"
    ]
    
    correct = 0
    total = 0
    
    for subject in subjects:
        ds = load_dataset("cais/mmlu", name=subject, split=f"validation[:{max_samples_per_subject}]")
        
        for example in ds:
            question = example["question"]
            choices = example["choices"]
            answer_idx = example["answer"]
            answer = ["a", "b", "c", "d"][answer_idx]

            prompt = (
                f"Question: {question}\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n"
                f"Answer:"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id
                )

            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            predicted_answer = extract_letter(prediction.replace(prompt, ""))

            is_correct = predicted_answer == answer
            correct += int(is_correct)
            total += 1

            print(f"\nQuestion: {question}")
            print(f"Choices: {choices}")
            print(f"Correct answer: {answer.upper()} | Model answer: {predicted_answer.upper()}")

    accuracy = (correct / total) * 100
    print(f"Accuracy {accuracy:.2f}%")

if __name__ == "__main__":
    base_model_id = "###"
    adapter_path = "###"

    model, tokenizer = load_model_local(base_model_id, adapter_path)
    evaluate_on_selected_mmlu(model, tokenizer, max_samples_per_subject=20)