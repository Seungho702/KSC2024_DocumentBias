import pandas as pd 
import time, argparse
from tqdm import tqdm
import csv
import os
from openai import OpenAI

###LLM이 media bias를 평가할 수 있는지를 확인하는 코드

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

few_shot_examples = """
Q: Labor Secretary Alex Acosta resigned amid the controversy surrounding his previous handling of the Jeffrey Epstein case. His departure comes as critics pointed to what they called a lenient plea deal Epstein received over a decade ago.
A: left

Q: The Knesset parliament of Israel enacted a controversial judicial overhaul, sparking protests across the country. Prime Minister Netanyahu defended the decision, saying it is necessary to curb the influence of the judiciary.
A: right

Q: Donald Trump stirred controversy at the weekend rally in Michigan, where he claimed that the 2020 election was "rigged." Despite multiple court rulings dismissing such claims, the former president's speech energized his base.
A: center

Q: Brits are bracing for record-breaking highs on Saturday, with temperatures set to soar above 40 degrees Celsius in some parts of the country. The UK Met Office has issued heat warnings across England.
A: right

Q: The Federal Reserve announced a 0.25% interest rate hike to curb inflation, citing ongoing concerns about price stability and the need to maintain economic growth in the long term.
A: center

Q: Vivek Ramaswamy has officially qualified for the upcoming Republican debate, as confirmed by the RNC this week. The biotech entrepreneur, running on a conservative populist platform, has continued to poll well among GOP voters.
A: left
"""

# Randomly sample 300 examples for each bias_rating ('left', 'right', 'center')
def sample_data(allsides_data, sample_size=300):
    left_samples = allsides_data[allsides_data['bias_rating'] == 'left'].sample(sample_size, random_state=42)
    right_samples = allsides_data[allsides_data['bias_rating'] == 'right'].sample(sample_size, random_state=42)
    center_samples = allsides_data[allsides_data['bias_rating'] == 'center'].sample(sample_size, random_state=42)
    combined_samples = pd.concat([left_samples, right_samples, center_samples])
    return combined_samples

def generate_gpt(args, client, text):
    cnt=0
    response = None
    model = 'gpt-4o' 
    while True:
        try:
            cnt+=1
            if cnt==5:
                break
            response = client.chat.completions.create(
                model=model,
                messages=[
                {"role": "system", "content": 
                """
                Please act as an impartial judge and rate the bias of the text data provided by news.
                Your evaluation should consider correctness.  

                Your job is to rate the text as right, left or center.
                Avoid any position biases and ensure that the order in which the responses were presented does not influence your
                decision. 
                Do not allow the length of the responses to influence your evaluation. 
                Do not favor certain names of the assistants.  

                Be as objective as possible. 
                After rating the text, output your final verdict by strictly following this format: 
                "[[right]]" if the text is right-biased, 
                "[[left]]" if the text is left-biased, 
                and "[[center]]" if the text is center.  

                DO NOT provide an explanation!!
                just provide [[right]] or [[left]] or [[center]]
                Be sure to include [[ ]].

                """},
                {"role": "user", "content": few_shot_examples + f"Q: {text}\nA:"}
            ],
            )
            break
                
        except Exception as e:
            print("Exception: ", e)
            print('print error message')
            time.sleep(10)
            
    if response == None:
        return None
    res = response.choices[0].message.content
    return res

def evaluate_model(args, client, combined_samples, output_file='Your output filename here'):
    c = 0
    total = len(combined_samples)
    results = []

    for idx, row in tqdm(combined_samples.iterrows(), total=total): 
        text = row['text']  
        original_bias = row['bias_rating']
        gpt_rating = generate_gpt(args, client, row['text'])
        results.append([text, gpt_rating, original_bias])
        if gpt_rating == f"[[{original_bias}]]":
            c += 1

    accuracy = c / total * 100
    print(f"Accuracy: {accuracy:.2f}% ({c}/{total})")

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text', 'gpt_rating', 'original_bias'])
        writer.writerows(results)  

    print(f"Results saved to {output_file}")  

args = argparse.Namespace(
    model='gpt-4o', 
    temperature=0,
    max_token_len=100,
    openai_api_key="Your API key here",
    api_key=0  
)

client = OpenAI(api_key="Your API key here")
combined_samples = sample_data(allsides_data)
evaluate_model(args, client, combined_samples, output_file='Your output filename here')