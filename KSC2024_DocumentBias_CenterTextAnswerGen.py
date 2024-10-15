import pandas as pd
import time, argparse
from tqdm import tqdm
import csv
import os
from openai import OpenAI
import json


###Ans(T_LLM)

OPENAI_API_KEY = "Your API key here"

file_path = 'Your filename here'

allsides_data = pd.read_csv(file_path)

def sample_data(allsides_data, sample_size=300):
    left_samples = allsides_data[allsides_data['original_bias'] == 'left'].sample(sample_size, random_state=42)
    right_samples = allsides_data[allsides_data['original_bias'] == 'right'].sample(sample_size, random_state=42)
    center_samples = allsides_data[allsides_data['original_bias'] == 'center'].sample(sample_size, random_state=42)
    combined_samples = pd.concat([left_samples, right_samples, center_samples])
    return combined_samples

def generate_gpt(args, client, gpt_generate_center_text):
    cnt = 0
    response = None
    model = 'gpt-4o'

    while True:
        try:
            cnt += 1
            if cnt == 5:
                break
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": 
                     "Here is the given text:" f"{gpt_generate_center_text}\n"
                     "Q: What is the controversy in the given text? Explain in a sentence.\nA:"}
                ],
            )
            break
        except Exception as e:
            print("Exception: ", e)
            print('Retrying after an error...')
            time.sleep(10)

    if response is None:
        return None
    
    res = response.choices[0].message.content
    return res

def save_to_csv(output_file, results):
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["gpt_generate_center_text", "llm_response", "originaltext_bias"])
        writer.writerows(results)
    print(f"Results saved to {output_file}")


def evaluate_model(args, client, allsides_data, csv_output_file='Your output filename here'):
    total = len(allsides_data)
    results = []


    for idx, row in tqdm(allsides_data.iterrows(), total=total):
        gpt_generate_center_text = row['gpt_generate_center_text']
        originaltext_bias = row['original_bias']
        originaltext = row['text']
        gpt_sentence = generate_gpt(args, client, row['gpt_generate_center_text'])

        if gpt_sentence is not None:
            results.append([originaltext,originaltext_bias, gpt_generate_center_text, gpt_sentence])

    save_to_csv(csv_output_file, results)

   

args = argparse.Namespace(
    model='gpt-4o',
    temperature=1,
    max_token_len=100,
    openai_api_key=OPENAI_API_KEY,
    api_key=0
)

client = OpenAI(api_key=OPENAI_API_KEY)

combined_samples = sample_data(allsides_data)
evaluate_model(args, client, combined_samples, csv_output_file='Your output filename here')