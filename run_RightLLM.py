import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ Model and tokenizer setup
cache_dir = "/data"
model_name = "SaisExperiments/Evil-Alpaca-Right-Lean-L3.2-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(device)

# ✅ CSV file paths
input_csv = "input_csv_file"
output_csv = "output_csv_file"

# ✅ Read CSV file
df = pd.read_csv(input_csv)

# ✅ Add LLM responses
responses = []

for question in df.iloc[:, 0]:  # Read the first column (questions)
    prompt = (
        f"Answer the following political question concisely and in detail: {question} "
        "Explain your reasoning and the principles underlying your views. Include supporting evidence or examples where relevant."
        "Do not repeat the question or include any section headers like '### Input:' or '### Response:'. "
        "Only output the answer."
        
    )
    # ✅ Generate response from LLM
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=500)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    responses.append(generated_text)  # Store LLM response

# ✅ Add the LLM responses to a new column and save
df["RightContext"] = responses
df.to_csv(output_csv, index=False)

print(f"✅ Response saved to: {output_csv}")
