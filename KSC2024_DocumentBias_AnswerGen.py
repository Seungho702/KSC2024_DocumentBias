import pandas as pd 
import time, argparse
from tqdm import tqdm
import csv
import json
import os
from openai import OpenAI
### allsides_data_before에서, 'left', 'center', 'right'가 모두 있는 data만 추려서 allsides_data에 저장
### LLM이 생성한 답변 Ans(T_bias)을 생성하는 코드

OPENAI_API_KEY = "Your API key here"

file_path = 'allsides_news.csv'

allsides_data_before = pd.read_csv(file_path)

valid_topics = (
    allsides_data_before.groupby('Topics')['bias_rating']
    .apply(lambda x: all(rating in x.values for rating in ['left', 'center', 'right']))
    .reset_index()
)

valid_topics = valid_topics[valid_topics['bias_rating'] == True]['Topics']
allsides_data = allsides_data_before[allsides_data_before['Topics'].isin(valid_topics)]

def sample_data(allsides_data, sample_size=300):
    left_samples = allsides_data[allsides_data['bias_rating'] == 'left'].sample(sample_size, random_state=42)
    right_samples = allsides_data[allsides_data['bias_rating'] == 'right'].sample(sample_size, random_state=42)
    center_samples = allsides_data[allsides_data['bias_rating'] == 'center'].sample(sample_size, random_state=42)
    combined_samples = pd.concat([left_samples, right_samples, center_samples])
    return combined_samples


def generate_gpt(args, client, text):
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
                     "Here is the given text:" f"{text}\n"
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
        writer.writerow(["text", "llm_response", "originaltext_bias"])
        writer.writerows(results)
    print(f"Results saved to {output_file}")

def save_to_json(output_file, results):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file}")


def evaluate_model(args, client, allsides_data, csv_output_file='Your output filename here', json_output_file='Your output filename here'):
    total = len(allsides_data)
    results = []
    json_results = []

    for idx, row in tqdm(allsides_data.iterrows(), total=total):
        text = row['text']
        originaltext_bias = row['bias_rating']
        gpt_sentence = generate_gpt(args, client, row['text'])

        if gpt_sentence is not None:
            results.append([text, gpt_sentence, originaltext_bias])
            context_id = f"context_{idx}"  
            result_data = {
                "context_id": context_id,
                "text": text,
                "llm_response": gpt_sentence,
                "eval": "TBD", 
                "context_bias": originaltext_bias
            }
            json_results.append(result_data)

    save_to_csv(csv_output_file, results)

    save_to_json(json_output_file, json_results)

args = argparse.Namespace(
    model='gpt-4o',
    temperature=1,
    max_token_len=100,
    openai_api_key=OPENAI_API_KEY,
    api_key=0
)

client = OpenAI(api_key=OPENAI_API_KEY)

combined_samples = sample_data(allsides_data)
evaluate_model(args, client, combined_samples, csv_output_file='Your output filename here', json_output_file='Your output filename here')

