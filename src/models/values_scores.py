import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the DataFrame
df = pd.read_csv('updated_application_data.csv')

# Define target words for each category
target_words = {
    "Accountable & Reflective": ["growth", "gratitude", "maturity", "integrity", "honesty"],
    "Motivated & Brave": ["achievement", "adventure", "initiative", "spark"],
    "Teamwork & Adaptability": ["dependable", "flexible", "consistent", "self-awareness"],
    "Open-Minded & Empathetic": ["respect", "decorum", "humility", "curious"],
    "Civic Duty & Historical Knowledge": ["community", "philanthropy", "well-read", "global-view"],
    "Career/Professional Knowledge & Employability": ["well-rounded", "professionalism", "literacy", "technology competence"]
}

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function to tokenize and truncate text
def tokenize_and_truncate(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)[:max_length-2]  # Reserve space for [CLS] and [SEP]
    return tokenizer.convert_tokens_to_string(tokens)

# # Function for sentiment analysis with tokenization, truncation, and confidence thresholds
# def get_sentiment_score(text, tokenizer):
#     truncated_text = tokenize_and_truncate(text, tokenizer)
#     results = sentiment_pipeline(truncated_text)
#     sentiment, confidence = results[0]['label'], results[0]['score']
    
#     # Define confidence thresholds
#     positive_threshold = 0.95  # Adjust based on your requirements
#     negative_threshold = 0.95  # Adjust based on your requirements
    
#     # Map sentiment to score based on confidence
#     if sentiment == 'POSITIVE' and confidence > positive_threshold:
#         return 3
#     elif sentiment == 'NEGATIVE' and confidence > negative_threshold:
#         return -1
#     else:
#         return 0  # Treat as neutral or uncertain sentiment

# Calculate average word count for each category
average_word_counts = {category: df[f'{category} Justification'].apply(lambda x: len(str(x).split())).mean() for category in target_words}

# Scoring functions for word count
def calculate_word_count_score(text, category):
    word_count = len(text.split())
    average_word_count = average_word_counts.get(category, 100)
    if word_count > average_word_count * 1.2:
        return 1
    elif word_count >= average_word_count * 0.8:
        return .5
    else:
        return 0

# Scoring function for target words
def calculate_target_words_score(text, category):
    score = sum(word in text.lower() for word in target_words.get(category, []))
    return min(score, 5)  # Maximum score for target words is 3

# Apply scoring to each category
for category in target_words.keys():
    justification_column = f'{category} Justification'
    word_count_column = f'{category} Word Count Score'
    # sentiment_column = f'{category} Sentiment Score'
    target_words_column = f'{category} Target Words Score'
    final_score_column = f'{category} Final Score'
    
    df[word_count_column] = df[justification_column].apply(lambda x: calculate_word_count_score(x, category))
    # df[sentiment_column] = df[justification_column].apply(lambda x: get_sentiment_score(x, tokenizer))
    df[target_words_column] = df[justification_column].apply(lambda x: calculate_target_words_score(x, category))
    df[final_score_column] = df[[word_count_column, target_words_column]].sum(axis=1) #sentiment_column,

# Save the updated DataFrame to a new CSV file
df.to_csv('final_scored_application_data_with_full_scoring.csv', index=False)
