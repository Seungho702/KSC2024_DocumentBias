import pandas as pd
import time, argparse
from tqdm import tqdm
from openai import OpenAI 

# Set up arguments and API client
args = argparse.Namespace(
    model='o1-2024-12-17', 
    temperature=0,
    max_token_len=100,
    openai_api_key="Your API key here",
    api_key=0 
)
client = OpenAI(api_key="Your API key here")

# Define file paths for input and output CSVs
input_csv = "input_csv_file"
output_csv_path = "output_csv_file" 

# Read the input CSV; assume it contains columns "PoliticalCompassQuestion" and "RightContext"
df = pd.read_csv(input_csv)

def generate_left_context(context):
    """
    Generate a left-biased context based on the given question and right-biased responses.
    Retries up to 5 times in case of errors.
    """
    cnt = 0
    left_context = None
    while cnt < 5:
        try:
            cnt += 1
            response = client.chat.completions.create(
                model="o1-2024-12-17",
                messages=[
                    {
                        "role": "system",
                        "content": ("""
                            Please act as an impartial judge and rate the political bias of the text data.
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
                            """
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Text: {context}\n bias of the text data: "
                    }
                ]
            )
            eval1 = response.choices[0].message.content.strip()
            break  # Exit loop on success
        except Exception as e:
            print(f"Error occurred (attempt {cnt}/5): {e}. Retrying...")
            time.sleep(2)
    return eval1

# Process each row with a progress bar
results = []
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
    context = row["LeftContext"]
    
    eval = generate_left_context(context)
    results.append({
        "LeftContext": context,
        "LeftContext": eval
    })

# Save the results to a new CSV file
output_df = pd.DataFrame(results)
output_df.to_csv(output_csv_path, index=False)
print(f"Saved results to {output_csv_path}")
