import os
import json
import openai
import pandas as pd

# Function to load configuration securely
def load_config(config_filename):
    with open(config_filename, "r") as file:
        return json.load(file)

# Define paths relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.json')
csv_path = os.path.join(script_dir, 'application_data.csv')

# Load configuration and set API key
config = load_config(config_path)
openai.api_key = config['dorrance_api']['api_key']

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Select only the columns that are marked as 'TRUE'
df = df[df['moving_forward']].head(1)

# Define the columns to analyze
columns_to_select = [2,4,6,9,10,11,12,13,14,15,16,18,19,21,22,24,25,27,28,30,31,33,34,35,37,38,40,41,43,44,46,47,49,50,52,53,54,56,57,59,60,62,63,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84]

# Select only the columns that are marked as 'TRUE'
df = df.iloc[:, columns_to_select]

# Dictionary of modifications where key is the iloc and value is the text to modify with
modification_dict = {
    0: 'My name is',
    1: 'I went to then high school',
    2: 'I want to attend the university',
    3: 'At university, I want to major in',
    4: 'After university, I want a career in',
    5: 'My matrix score is',
    6: 'My gpa is',
    7: 'My SAT combined score is',
    8: 'My ACT composite score is',
    9: 'An extracuriclor activity I participated in during high school was',
    10: 'And I accomplished',
    11: 'An extracuriclor activity I participated in during high school was',
    12: 'And I accomplished',
    13: 'An extracuriclor activity I participated in during high school was',
    14: 'And I accomplished',
    15: 'An extracuriclor activity I participated in during high school was',
    16: 'And I accomplished',
    17: 'An extracuriclor activity I participated in during high school was',
    18: 'And I accomplished',
    19: 'An extracuriclor activity I participated in during high school was',
    20: 'And I accomplished',
    21: 'Particpating in these extracurricular activities has prepared me for college by',
    22: 'A volunteer activity I participated in while in high school was',
    23: 'And I accomplished',
    24: 'A volunteer activity I participated in while in high school was',
    25: 'And I accomplished',
    26: 'A volunteer activity I participated in while in high school was',
    27: 'And I accomplished',
    28: 'A volunteer activity I participated in while in high school was',
    29: 'And I accomplished',
    30: 'A volunteer activity I participated in while in high school was',
    31: 'And I accomplished',
    32: 'A volunteer activity I participated in while in high school was',
    33: 'And I accomplished',
    34: 'The one volunteer experience that mattered to me most was',
    35: 'In high school I had a job as a',
    36: 'And worked at a company named',
    37: 'In high school I had a job as a',
    38: 'And worked at a company named',
    39: 'In high school I had a job as a',
    40: 'And worked at a company named',
    41: 'In high school I had a job as a',
    42: 'And worked at a company named',
    43: 'The university is a good fit for me because',
    44: 'My college minor is',
    45: 'I chose this college major because',
    46: 'My most prized posession is',
    47: 'My favorite author is',
    48: 'The thing I will miss most about leaving high school is',
    48: 'The most imporant thing I learned in high school was',
    50: 'The thing I am most looking forward to in college is',
    51: 'The thing I am most optimistic about our world today is',
    52: 'My favorite thing about arizona is',
    53: 'The feeling I get when I think about traveling abroad is',
    54: 'Out of the options funny, smart, or rich I would rather be',
    55: 'When I need advice I ask',
    56: 'The thing that excites me about being at a university with college students from all over the world?',
    57: 'My curiosity was sparked and allowed me to delve deeply into the subject when',
    58: 'A question I would you like to ask the instructor about free speech is',
    59: 'Something that significantly influenced my worldview was',
    60: 'My uprbining was shaped by',
    61: 'A time when I had to choose between what is right versus what was best for myself was',
    62: 'My matrix total is'
}

# Assume df.columns are integer-indexed or you're working with position-based modification
for index, text in modification_dict.items():
    if index in range(len(df.columns)):  # Checks if the index is within the DataFrame's column range
        df.iloc[:, index] = df.iloc[:, index].apply(lambda x: text + " " + str(x) if pd.notna(x) and x != '' else x)

def combine_text(row, columns, df_shape):
    valid_columns = [col for col in columns if col < df_shape[1]]  # Ensure column index is within bounds
    return ' '.join(str(row[col]) for col in valid_columns if not pd.isnull(row[col]))

# Update the application of the combine_text function, pass the current DataFrame shape
df['combined_text'] = df.apply(lambda row: combine_text(row, columns_to_select, df.shape), axis=1)

# Updated dictionary of prompts for each value category
values_prompts = {
    "Accountable & Reflective": "Analyze an applicant's submission for a scholarship organization seeking first-generation college students. Summarize how the applicant demonstrates accountability and thoughtfulness. Use specific examples and quotes to support your summary. If the applicant's examples do not convincingly display these qualities, or if there are mixed sentiments, accurately reflect this in your summary. '{}'.",
    "Motivated & Brave": "Review an applicant's submission for a scholarship aimed at first-generation college students. Describe any instances where the applicant shows motivation and bravery. Include specific examples and quotes from their submission. If the submission lacks strong evidence of these qualities, or if it reveals both strengths and weaknesses in these areas, your summary should reflect the actual sentiment. '{}'.",
    "Teamwork & Adaptability": "Examine an applicant's submission for a scholarship designed for first-generation college students. Highlight instances of teamwork and adaptability. Provide specific examples and quotes from the submission. If the submission does not clearly showcase these qualities, or if it includes both positive and negative aspects, ensure your summary accurately represents this sentiment. '{}'.",
    "Open-Minded & Empathetic": "Analyze a scholarship applicant's submission focused on first-generation college students. Summarize how the applicant displays open-mindedness and empathy through specific examples and quotes. If the submission lacks convincing evidence or shows a range of sentiments regarding these qualities, your summary should accurately reflect these findings. '{}'.",
    "Civic Duty & Historical Knowledge": "Review a scholarship applicant's submission for first-generation college students. Detail the applicant's community involvement and historical knowledge using specific examples and quotes. If the applicant's submission does not convincingly demonstrate these qualities, or if it includes both positive and negative sentiments, reflect this accurately in your summary. '{}'.",
    "Career/Professional Knowledge & Employability": "Examine a scholarship applicant's submission targeting first-generation college students. Describe how the applicant showcases their career or professional knowledge with specific examples and quotes. If the submission lacks strong examples of these qualities, or if it presents a mix of strengths and weaknesses, ensure your summary reflects the true sentiment. '{}'.",
}

# Initialize justifications columns in the DataFrame
for value in values_prompts.keys():
    df[f'{value} Justification'] = ''

def get_openai_summary_for_justifications(text, prompt_template):
    # Adjusted to use the chat completions endpoint for GPT-4 Chat models
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",  # Ensure this is the correct model name for GPT-4 Chat
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt_template.format(text)}],
        temperature=0.7,  # Adjust for predictability
        # max_tokens=500,  # Control output length for uniformity
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Assuming the structure of the response object fits, adjust as necessary
    return response['choices'][0]['message']['content'].strip()

def synthesize_cohesive_summary(individual_summaries, candidate_name):
    # Personalize the summary by starting with the candidate's name
    personalized_intro = f"{candidate_name} is a scholarship candidate who"
    # Combine individual summaries into a single text for synthesis
    combined_text = " ".join([personalized_intro] + individual_summaries)
    synthesis_prompt = (
        "Given the individual summaries below, create a cohesive and concise narrative that integrates all key points seamlessly. "
        "Avoid repetition and ensure the narrative flows logically, capturing the essence of the applicant's qualifications and potential across all value types. '{}'"
    ).format(combined_text)

    # Call the API or summarization function for synthesis
    cohesive_summary = get_openai_summary_for_justifications(combined_text, synthesis_prompt)
    return cohesive_summary

def summarize_justifications(row, values_prompts_keys):
    individual_summaries = []  # List to hold summaries for each value type
    for value in values_prompts_keys:  # Directly iterate over the keys
        justification_text = row[f'{value} Justification']
        # Ensure the justification text is correctly passed to the summary function
        if justification_text.strip() != '':
            summary = get_openai_summary_for_justifications(justification_text, values_prompts[value])
            individual_summaries.append(summary)  # Add to the list
        else:
            individual_summaries.append(f"No data provided for {value}.")  # Placeholder if no justification text

    # Synthesize individual summaries into a cohesive narrative
    candidate_name = row['Full Name']  # Adjust this to match your DataFrame structure for candidate names
    final_summary = synthesize_cohesive_summary(individual_summaries, candidate_name)
    return final_summary

# Analyze each relevant field for each candidate using the 'combined_text' column
for index, row in df.iterrows():
    combined_text = row['combined_text']

    for value, prompt in values_prompts.items():
        justification = get_openai_summary_for_justifications(combined_text, prompt)

        # Update the DataFrame with the justification only
        df.loc[index, f'{value} Justification'] = justification

# Add a new column 'value_prompts_summarized' for the summarized justifications
df['value_prompts_summarized'] = df.apply(lambda row: summarize_justifications(row, list(values_prompts.keys())), axis=1)

# Export the DataFrame to a new CSV, adjust column indices as needed
output_csv_path = os.path.join(script_dir, 'application_data_updated.csv')
df.to_csv(output_csv_path, index=False)