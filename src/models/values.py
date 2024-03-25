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
df = df[df['moving_forward']].head(10)

# Define the columns to analyze
columns_to_select = [2,4,6,9,10,11,12,13,14,15,16,18,19,21,22,24,25,27,28,30,31,33,34,35,37,38,40,41,43,44,46,47,49,50,52,53,54,56,57,59,60,62,63,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84]

# Select only the columns that are marked as 'TRUE'
df = df.iloc[:, columns_to_select]

# Dictionary of modifications where key is the iloc and value is the text to modify with
modification_dict = {
    0: 'My name is',
    1: 'I went to the high school',
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
    61: 'A time when I had to choose between what is right versus what was best for myself was'
    # 62: 'My matrix total is'
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
    "Accountable & Reflective": "Create a concise summary of this scholarship applicant in narrative form, focusing on pivotal examples and quotes that demonstrate accountability and reflectiveness. Aim to highlight the most compelling evidence of these qualities, ensuring a balanced portrayal if both strengths and weaknesses are apparent. '{}'.",
    "Motivated & Brave": "Create a concise summary of this applicant in narrative form, focusing on pivotal examples and quotes that demonstrate motivation and bravery. Aim to highlight the most compelling evidence of these qualities, ensuring a balanced portrayal if both strengths and weaknesses are apparent. '{}'.",
    "Teamwork & Adaptability": "Create a concise summary of this applicant in narrative form, focusing on pivotal examples and quotes that demonstrate teamwork and adaptability. Aim to highlight the most compelling evidence of these qualities, ensuring a balanced portrayal if both strengths and weaknesses are apparent. '{}'.",
    "Open-Minded & Empathetic": "Create a concise summary of this applicant in narrative form, focusing on pivotal examples and quotes that demonstrate open-mindedness and empathy. Aim to highlight the most compelling evidence of these qualities, ensuring a balanced portrayal if both strengths and weaknesses are apparent. '{}'.",
    "Civic Duty & Historical Knowledge": "Create a concise summary of this applicant in narrative form, focusing on pivotal examples and quotes that demonstrate civic duty and historical knowledge. Aim to highlight the most compelling evidence of these qualities, ensuring a balanced portrayal if both strengths and weaknesses are apparent. '{}'.",
    "Career/Professional Knowledge & Employability": "Create a concise summary of this applicant in narrative form, focusing on pivotal examples and quotes that demonstrate professionalism and employability. Aim to highlight the most compelling evidence of these qualities, ensuring a balanced portrayal if both strengths and weaknesses are apparent. '{}'.",
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
        max_tokens=500,  # Control output length for uniformity
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Assuming the structure of the response object fits, adjust as necessary
    return response['choices'][0]['message']['content'].strip()

def generate_interview_questions(summary):

    prompt = (
        f"Based on the candidates application summary, generate three concise, direct questions for a follow-up interview. "
        f"These questions should aim to delve deeper into the candidate's experiences, achievements, and motivations, "
        f"without being overly descriptive or verbose. Keep the questions straightforward and focused. Summary: '{summary}'"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",  # Ensure this model name aligns with your use case
            messages=[
                {"role": "system", "content": "You are a helpful assistant tasked with generating concise, insightful interview questions based on application summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        generated_text = response.choices[0].message['content'].strip()

        # Processing the generated text to extract and clean up the questions
        questions = generated_text.split('\n')
        questions = [question.strip() for question in questions if question.strip()]

        return questions[:3]  # Returning the first three questions for brevity and focus
    except Exception as e:
        print(f"Failed to generate interview questions: {e}")
        return ["An error occurred while generating interview questions."]

def synthesize_cohesive_summary(individual_summaries, candidate_name, interview_questions):
    personalized_intro = f"{candidate_name} is a scholarship candidate who"
    combined_text = " ".join([personalized_intro] + individual_summaries)
    
    synthesis_prompt = (
        "Create a summary that maintains a realistic and neutral tone. "
        "The summary should be clear and concise, avoiding the use of overly descriptive language or an abundance of adjectives. "
        "Focus on portraying the applicant's qualifications, experiences, and aspirations accurately, without embellishment. "
        "Present the applicant's story in a way that is factual and objective, ensuring that the narrative is grounded in the applicant's actual accomplishments and stated goals. '{}'"
    ).format(combined_text)

    cohesive_summary = get_openai_summary_for_justifications(combined_text, synthesis_prompt)

    # Construct the final narrative with the formatted questions
    final_narrative = f"{cohesive_summary}\n\nTo delve deeper into their experiences and aspirations, consider asking the following questions during the interview:"
    
    for i, question in enumerate(interview_questions, start=1):
        final_narrative += f"\nQuestion {question}"

    return final_narrative

def summarize_justifications(row, values_prompts_keys):
    individual_summaries = []  # List to hold summaries for each value type
    for value in values_prompts_keys:  # Directly iterate over the keys
        justification_text = row[f'{value} Justification']
        if justification_text.strip() != '':  # Check if justification text is not empty
            summary = get_openai_summary_for_justifications(justification_text, values_prompts[value])
            individual_summaries.append(summary)  # Add to the list
        else:
            individual_summaries.append(f"No data provided for {value}.")  # Placeholder if no justification text

    # Synthesize individual summaries into a cohesive narrative
    candidate_name = row['Full Name']  # Adjust to match your DataFrame structure for candidate names
    
    # Generate interview questions based on the combined text from individual summaries
    interview_questions = generate_interview_questions(" ".join(individual_summaries))
    
    # Now call synthesize_cohesive_summary with all required arguments
    final_summary = synthesize_cohesive_summary(individual_summaries, candidate_name, interview_questions)
    
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