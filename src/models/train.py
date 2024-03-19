import os
import json
import openai
import pandas as pd

# Set your OpenAI API key
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
config_file = open(config_path, "r")
config = json.load(config_file)

openai.api_key = config['dorrance_api']['api_key']

# Load the CSV file into a DataFrame
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'application_data.csv')
df = pd.read_csv(csv_path).head(10)

individual_columns = [33, 52]
range_of_columns = list(range(65, 83))  # 84 is not included
columns_to_select = individual_columns + range_of_columns

df_selected = df.iloc[:, columns_to_select]

# A function to get a score and summary justification from the OpenAI API
def get_openai_score_and_justification(text, prompt_template):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_template.format(text=text),
        temperature=0.5
    )
    return response.choices[0].text.strip()

# Define a dictionary of prompts for each value category
values_prompts = {
    # For each value, create a prompt that instructs the AI to provide a score and a justification.
    "Accountable & Reflective": "Consider the following description of activities and achievements: '{}'. On a scale of 1 to 10, rate how well this reflects accountability and thoughtfulness, and provide a brief justification for your score.",
    # Add similar prompt templates for the other values...
}

# Initialize columns for scores and justifications in the DataFrame
for value in values_prompts.keys():
    df[f'{value} Score'] = 0
    df[f'{value} Justification'] = ''

# Analyze each relevant field for each candidate
for index, row in df.iterrows():
    # Concatenate all text fields you want to analyze for each candidate
    combined_text = f"{row['Activity 1 | Accomplishments']} {row['Volunteer Activity 1 | Accomplishments']} {row['Short Answer Question 1']}"

    # Iterate over each value and prompt
    for value, prompt in values_prompts.items():
        # Get the OpenAI API analysis for the text
        analysis_result = get_openai_score_and_justification(combined_text, prompt)
        
        # Here, you'll parse the analysis_result to extract both the score and the justification
        # This parsing depends on how you've structured your prompts and the kind of responses you get from the API
        # For example, if your response is "The score is 7 out of 10 because the candidate shows strong reflection in their activities."
        # You would parse the "7" as the score and the rest as the justification
        score = ... # extract the score
        justification = ... # extract the justification
        
        # Update the DataFrame with the score and the justification
        df.loc[index, f'{value} Score'] = score
        df.loc[index, f'{value} Justification'] = justification

# After processing all candidates, export the DataFrame to a new CSV
df.to_csv('path_to_your_output_file.csv', index=False)