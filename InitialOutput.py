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

# ✅ Read CSV file (expecting columns: 'PoliticalCompassQuestion' and 'LeftContext')
df = pd.read_csv(input_csv)

# ✅ Generate responses using the context and question
responses = []

for idx, row in df.iterrows():
    question = row['PoliticalCompassQuestion']
    
    
    # Create a chat conversation with special tokens using system and user messages
    chat = [
        {"role": "system", "content": "Based on your knowledge, please answer the following question."},
        {"role": "user", "content": f"Question: {question}\nYour answer must be one of the following options only: Strongly disagree, Disagree, Agree, Strongly agree."}
    ]
    
    # Apply the chat template to insert special tokens (returning a string)
    chat_str = tokenizer.apply_chat_template(chat, tokenize=False)
    
    # Tokenize the resulting string with return_tensors="pt"
    inputs = tokenizer(chat_str, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response using a very small temperature (near-deterministic sampling)
    output = model.generate(**inputs, max_length=3000, temperature=1e-8, top_p=1, do_sample=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    responses.append(generated_text.strip())

# ✅ Save the LLM responses in a new column called 'FinalOutput1' and export to CSV
df["FinalOutput1"] = responses
df.to_csv(output_csv, index=False)

print(f"✅ Response saved to: {output_csv}")
