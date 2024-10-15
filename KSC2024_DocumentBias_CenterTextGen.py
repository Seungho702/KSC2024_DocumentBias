import pandas as pd 
import time, argparse
from tqdm import tqdm
import csv
import os
from openai import OpenAI
### 기존 텍스트 T_bias를 중립적인 텍스트 T_LLM으로 변환하는 코드

OPENAI_API_KEY = "Your API key here"

file_path = 'Your filename here'

orginal_text_data = pd.read_csv(file_path)


def generate_gpt(args, client, text):
    cnt=0
    response = None

    ##gpt-4o-2024-05-13
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
                
                Your job is to change the given text from right or left into center.
                
                "Center" means DO NOT contain Left or Right features.
                
                Maintain the main event of the given text.
                
                Please avoid including the "Left" or "Right" features.
                
                I'll tell you about the features of the Left and Right text below.
                
                "Left" text displays media bias in ways that strongly align with liberal, progressive, or left-wing thought and/or policy agendas.
                Texts with a Left media bias are most likely to show favor for:
                    Generous government services (food stamps, social security, Medicare, student-loans, unemployment benefits, healthcare, education, etc.)
                    A rejection of social and economic inequality
                    A belief in systemic oppression and a need for the government to step in and rectify the wrongs it has committed
                    Federal laws to protect consumers and the environment
                    Federal laws against discrimination
                    Federal laws protecting equal rights
                    Tax increases on the wealthy
                    Government regulation of corporations
                    Keeping abortion legal and accessible
                    A belief that some groups of people suffer disproportionately greater amounts in society due to identity characteristics, including race, gender, sexual orientation, or religion
                    Decreasing military spending and intervention
                    A belief in individualism and the protection of personal freedoms
                    A belief in generous immigration policies
                    A belief that the role of government is not just to protect rights, but to provide for its people and end suffering
                    A belief that government should prevent wealth from concentrating in the hands of a few
                    A belief that all humans have a right to healthcare, housing, clean water, a living wage
                    A belief that all people deserve help when they have fallen on hard times
                    An embrace of empathy, compassion, and tolerance as guiding values
                    A belief in the importance of multiculturalism and representation of diverse cultures and races in media, positions of political power, and corporations
                    Concerns about hate speech
                    A belief in “live and let live,” i.e, that the government should not intervene just because someone is acting in ways someone else does not approve of, provided they have harmed no one else
                    A belief that corporations, if left unregulated, may do harm to workers, society and the environment in the pursuit of profit
                    
                    

                "Right" text displays media bias in ways that strongly align with conservative, traditional, or right-wing thought and/or policy agendas.
                Texts with a Right media bias are most likely to show favor for:
                    Freedom of speech
                    Traditional family values
                    Decreasing taxes
                    Preserving the rights of gun owners
                    Outlawing or restricting abortion
                    Reliance on personal responsibility rather than government fiat
                    Decreasing federal regulations, giving more power to state laws
                    Decreasing government spending and involvement in economic issues
                    Preserving the philosophy and rules enshrined in the U.S. Constitution
                    Rejection of total equality or equity as an organizing principle of society
                    Belief in equality under the law and equal opportunity, with no favoritism, subsidies, or targeted prohibitions imposed by government
                    Belief in the sovereignty of the individual over the collective and the preservation of all personal freedoms (libertarian thought)
                    Belief that some personal freedoms may need to be limited (such as drug use) to maintain public order and societal flourishing (conservative thought)
                    Belief that government should be as small and non-intrusive as possible, leaving individuals to make their own decisions (libertarian thought)
                    Belief that government should encourage decisions that lead to societal flourishing (such as family formation) and discourage harm (conservative thought)
                    Rejection of left-wing identity politics, gender identity, affirmative action, the “welfare state”
                    Maintaining strong border security; ensuring all immigrants enter through a legal process or restricting immigration entirely
                    Rejection of laws that impose unnecessary burdens on businesses/the economy
                    Belief that the government needs to provide some collective goods (water, public parks, libraries) but its scope should be very limited
                    Belief that tradition and prevailing cultural norms contain wisdom that has been handed down and should be preserved
                    Preserving a traditional moral framework, often as outlined in religious traditions, through the use of state laws
                    Balanced government budgets and fiscal conservatism
                    

                """},
                {"role": "user", "content": f"here is the given text: {text}\nA:"}
             
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
    
    total = len(combined_samples)

    results = []

    for idx, row in tqdm(combined_samples.iterrows(), total=total): 
        text = row['text']  
        original_bias = row['originaltext_bias']

        gpt_generate_center_text = generate_gpt(args, client, row['text'])

        results.append([text, original_bias, gpt_generate_center_text])


    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text', 'original_bias', 'gpt_generate_center_text'])  
        writer.writerows(results) 

    print(f"Results saved to {output_file}")  


args = argparse.Namespace(
    model='gpt-4o', 
    temperature=1,
    max_token_len=102,
    openai_api_key="Your API key here",  
    api_key=0  
)

client = OpenAI(api_key="Your API key here")
combined_samples = orginal_text_data
evaluate_model(args, client, combined_samples, output_file='Your output filename here')