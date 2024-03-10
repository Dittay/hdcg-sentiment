# ========= Capstone Project =========

# Import modules
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import os

# Ask user to confirm absolute path of the .csv file
# Use os module to handle this

# Ask user to confirm the file path
file_path = input("Please enter the path to the amazon_product_reviews.csv file (make sure to include the file itself):\n")

# Ensure provided file path exists and is a file
if os.path.exists(file_path) and os.path.isfile(file_path) and file_path.endswith('.csv'):
    # Read CSV file
    try:
        df = pd.read_csv(file_path)
        print("File read successfully.")

    # Otherwise handle the error
    except Exception as e:
        print("An error occured trying to read the file:", e)

# Handle other edge cases
else:
    print("The filepath does not exists or is not a file")


# Load NLP model
nlp = spacy.load('en_core_web_sm')

# Add SpacyTextBlob to the pipeline
nlp.add_pipe('spacytextblob')

# Assign column of interest to variable
reviews_data = df['reviews.text']

# Assuming there were missing values, drop these
clean_data = df.dropna(subset=['reviews.text'])

# Text pre-processing - Function works on specific column in wider dataframe
# To have it work on a dataframe that is only one column, or generally any column, replace x = x with x.iloc[:, 0] = x.iloc[:, 0]

def text_clean(x):
    # Set to lower case
    x = x.str.lower()

    # Remove one or more whitespace characters including other unicode ones
    x = x.str.replace(r'\s+', ' ', regex=True)

    # Remove set of special characters
    remove_spec_chars = ["!",'"',"%","&","'","(",")","#","*","?",
                    "+",",","-","/",":",";","<","=",">",
                    "@","[","\\","]","^","_","`","{","|","}",
                    "~","–","’", "*"]
    
    for char in remove_spec_chars:
        x = x.str.replace(char, ' ')

    # Handle periods not part of abbreviations
    # This pattern aims to remove periods that are not followed by a lowercase letter (common in abbreviations)
    x = x.str.replace(r'\.(?![a-z])', ' ', regex=True)

    # Remove single characters
    x = x.str.replace(r'\b[a-zA-Z]\b', ' ', regex=True)

    # Remove extra spaces (trim) from both ends
    x = x.str.strip()

    # Remove double spacing
    x = x.str.replace(r' +', ' ', regex=True)

    # Remove spaces --
    x = x.str.replace(r'--', '', regex=True)

    return x

# Apply function to clean data variable
clean_data = text_clean(clean_data['reviews.text'])

# Create a function that takes in a product review as input and predicts its sentiment

def predict_sentiment(review_text, review_index):
    # Process review using the model
    doc = nlp(review_text)

    # Determine polarity and subjectivity using SpacyTextBlob
    polarity = doc._.polarity
    subjectivity = doc._.subjectivity

    # Determine sentiment using polarity
    if polarity > 0:
        sentiment = 'positive'
    elif polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return sentiment, polarity, subjectivity, review_index

# Ask user to pick a review
user_review_choice = int(input(f"Please choose a review using a value from 0 to {(reviews_data.size)-1} (please only enter whole numbers)\n"))

# Assign review choice to index variable
review_index = user_review_choice
review_of_choice = clean_data[review_index]

# Apply function to chosen review to conduct sentiment analysis
sentiment, polarity, subjectivity, review_index = predict_sentiment(review_of_choice, review_index)

# Print resuts
print(f"Review index: {review_index}\nSentiment: {sentiment}, Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}")