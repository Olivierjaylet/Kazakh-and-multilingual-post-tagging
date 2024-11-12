from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score


import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

import os



def data_to_nltk(df):
    # Convert the data into the format that NLTK expects (list of tuples)
    tagged_sentences = []
    sentence = []

    for _, row in df.iterrows():
        if row['WORD'] == ".":  # End of a sentence (you may need to adjust this)
            sentence.append((row['WORD'], row['POS']))
            tagged_sentences.append(sentence)
            sentence = []
        else:
            sentence.append((row['WORD'], row['POS']))

    # Handle any remaining sentence
    if sentence:
        tagged_sentences.append(sentence)
    return tagged_sentences

def extract_words_and_tags(nested_list):
    # Flatten the nested list of tuples
    words = [word for sentence in nested_list for word, _ in sentence]
    tags = [tag for sentence in nested_list for _, tag in sentence]
    
    # Convert the lists to numpy arrays
    words_array = np.array(words, dtype=object)
    tags_array = np.array(tags, dtype=object)
    
    return words_array, tags_array

# Function to extract words and POS tags for classification report
def extract_tags(tagged_data, tagger):
    y_true = []
    y_pred = []
    for sentence in tagged_data:
        words, true_tags = zip(*sentence)  # separate words and tags
        predicted_tags = []
        
        # Predict tags, handling unknown tags
        for word in words:
            
            prediction = tagger.tag([word])[0][1] if tagger.tag([word]) else "UNK"
            predicted_tags.append(prediction)
        y_true.extend(true_tags)
        y_pred.extend(predicted_tags)
    
    return y_true, y_pred