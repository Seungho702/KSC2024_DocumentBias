import pandas as pd
import time, argparse
from tqdm import tqdm
from openai import OpenAI  

# Set up arguments and API client
args = argparse.Namespace(
    model='gpt-4o', 
    temperature=1,
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

def generate_Right_context(question):
   
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
                        "content": """
  
                             You are an expert political analyst specializing in conservative, traditional, and right-wing thought. When you receive a question, your task is not simply to answer it directly, but to first think a comprehensive context that can be used as a reference for crafting your final answer. Follow these guidelines when creating the context:
                             
                            Identify the key topics, issues, and themes presented in the question.
  
                            Create a detailed background framework that highlights conservative values and policies relevant to the question. Refer to:
                            - Freedom of speech and the protection of individual liberties.
                            - Traditional family values and the importance of preserving established moral frameworks.
                            - Emphasis on personal responsibility over government intervention.
                            - Policies focused on decreasing taxes, reducing government spending, and limiting federal regulations.
                            - Preservation of constitutional rights, including the rights of gun owners.
                            - Advocacy for restricting or outlawing abortion in order to protect unborn life.
                            - Strong border security and controlled immigration to preserve national sovereignty.
                            - A commitment to fiscal conservatism and balanced government budgets.
                            
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
                            
                            Use current examples, historical context, and relevant data where applicable to illustrate how these right-wing principles apply to the question at hand.
                            Ensure that the generated context offers a rich, informative reference that can later be used to build a final answer. Do not provide the final answer now—focus solely on creating this foundational context.
                            Your generated context should serve as a robust, right-biased reference point to guide the final answer, ensuring that all responses are informed by conservative values and thorough analysis.

                            Do not include any additional content; only generate a Right-Biased Context.
                            
                            DO NOT INCLUDE ANY ADDITIONAL CONTENT; ONLY GENERATE A RIGHT-BIASED CONTEXT.
                        """
                                                
                    },
                    {
                        "role": "user",
                        "content": f"the question is: {question}\nRight-biased context: "
                    }
                ]
            )
            Right_context = response.choices[0].message.content.strip()
            break  # Exit loop on success
        except Exception as e:
            print(f"Error occurred (attempt {cnt}/5): {e}. Retrying...")
            time.sleep(2)
    return Right_context

# Process each row with a progress bar
results = []
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
    question = row["PoliticalCompassQuestion"]
    Right_context = generate_Right_context(question)
    results.append({
        "PoliticalCompassQuestion": question,
        "LeftContext": Right_context
    })

# Save the results to a new CSV file
output_df = pd.DataFrame(results)
output_df.to_csv(output_csv_path, index=False)
print(f"Saved results to {output_csv_path}")

