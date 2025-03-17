import pandas as pd
import time, argparse
from tqdm import tqdm
from openai import OpenAI  

# Set up arguments and API client
args = argparse.Namespace(
    model='gpt-4o', 
    temperature=0,
    max_token_len=100,
    openai_api_key="Your API key here",
    api_key=0 
)
client = OpenAI(api_key="Your API key here")

# Define file paths for input and output CSVs
input_csv = "/data/shcho/PoliticalCompass4LeftContext.csv"
output_csv_path = "/data/shcho/LeftContext.csv"

# Read the input CSV; assume it contains columns "PoliticalCompassQuestion" and "RightContext"
df = pd.read_csv(input_csv)

def generate_left_context(question, responses):
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
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert political analyst who generates left-biased contexts as counterarguments to right-biased responses.\n"
                            "Given a question and its associated right-biased responses, you must produce one left-biased context that meets the following requirements:\n"
                            "1. The subject of your output must match the subject discussed in the right-biased responses.\n"
                            "2. The context must be directly relevant to the given question and politically contrast with the right-biased responses.\n"
                            "3. For the given question, generate a left-biased context as if it were produced by someone whose political views are the exact opposite of those expressed in the right-biased responses.\n"
                            "4. The left-biased context must include logical reasoning and supporting evidence.\n"
                            "5. Avoid simply adding negations or superficially substituting right-biased terms with left-biased ones.\n"
                            "6. Even if multiple right-biased responses are provided, produce only one left-biased context.\n"
                            "7. If any of the right-biased responses include nonsensical or absurd content, you may disregard that content.\n\n"
                            "When a user submits a query formatted as:\n\n"
                            "Q: <question>, Right-biased responses: <responses>\n\n"
                            "your output should consist solely of the left-biased context. Do not repeat the question or include any additional commentary."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Q: {question}\nRight-biased responses: {responses}"
                    }
                ]
            )
            left_context = response.choices[0].message.content.strip()
            break  # Exit loop on success
        except Exception as e:
            print(f"Error occurred (attempt {cnt}/5): {e}. Retrying...")
            time.sleep(2)
    return left_context

# Process each row with a progress bar
results = []
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
    question = row["PoliticalCompassQuestion"]
    responses = row["RightContext"]
    left_context = generate_left_context(question, responses)
    results.append({
        "PoliticalCompassQuestion": question,
        "LeftContext": left_context
    })

# Save the results to a new CSV file
output_df = pd.DataFrame(results)
output_df.to_csv(output_csv_path, index=False)
print(f"Saved results to {output_csv_path}")
