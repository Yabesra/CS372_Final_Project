# src/inference.py – distilgpt2 optimized for yes/no
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#chose a smaller model for faster inference (distilgpt2 is a distilled version of gpt2, optimized for speed and smaller size)
#Phi‑1.5 is better for accuracy, but very slow
MODEL_NAME = "distilgpt2"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("Model ready.")

def rag_answer(query, context, max_new_tokens=10):
    """Ask a yes/no question and return answer (expects 'yes' or 'no')."""
    prompt = f"Question: {query}\nContext: {context}\nAnswer (yes or no):"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the answer after the prompt
    if "Answer (yes or no):" in response:
        response = response.split("Answer (yes or no):")[-1].strip()
    return response.lower()