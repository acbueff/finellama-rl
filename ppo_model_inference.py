
import os
import json
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def create_prompt(question: str) -> str:
    return f"Solve the following grade school math problem step by step:\n{question}\n\nSolution:"

def main():
    # Parse arguments
    model_id = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    max_tokens = int(sys.argv[4])
    use_4bit = sys.argv[5].lower() == 'true'
    
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # Load base model and tokenizer
    base_model_id = "SunnyLin/Qwen2.5-7B-DPO-VP"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    # Load model
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        adapter_path = model_id
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        adapter_path = model_id
        model = PeftModel.from_pretrained(model, adapter_path)
    
    # Generate predictions
    predictions = []
    for question in questions:
        prompt = create_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predictions.append(response)
    
    # Save predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
