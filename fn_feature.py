from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score


import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

import os



###################################################################################################
####################################### FEATURE ENGINEERING #######################################
###################################################################################################


def list_all_POS_tags(
        y
        ) :
    """Lists all unique POS tags from the input array.

    Args:
        y (array-like): Array of POS tag strings.

    Returns:
        list: A list of unique POS tags.
    """

    list_tags = []
    for tag_list in y :
        tags = tag_list.split()
        for tag in tags :
            if tag not in list_tags :
                list_tags.append(tag)
    return list_tags

def clean_data(
        df
        ):
    # Drop rows where 'WORD' is NaN
    df = df.dropna(subset=['WORD'])

    # Remove punctuation characters
    #df = df[df['POS'] != 'PUNCT']
    df = df[(df['POS'] != 'PUNCT') | (df['WORD'] == '.')]
    # Some characters to remove
    #chars_to_remove = r"[\#\$\%\&\(\)\+\:\@]"
    chars_to_remove = r"[\#\$\%\&\(\)\+\,\-\–\’\:\@\']"

    # Removing the characters from the 'WORD' column
    df['WORD'] = df['WORD'].str.replace(chars_to_remove,
                                        '',
                                        regex=True
                                        )
    
    # list of tags we want to predict
    POS_tag_keep = ['NOUN', 'VERB', 'ADJ', 'PROPN', 'NUM', 'ADV', 'ADP', 'CCONJ', 'PRON', 'DET', 'SCONJ', 'AUX', 'INTJ', 'PUNCT']
    
    df = df[df['POS'].isin(POS_tag_keep)]

    # convert to string type
    df["WORD"] = df["WORD"].astype(str)
    df["POS"] = df["POS"].astype(str)

    df = df.head(n=20000
                )

    print("Size dataset : ", df.shape)

    return df


def get_values(df_) :
    X_lex = df_['WORD'].str.strip()
    X_lex = X_lex.values

    y_lex = df_['POS'].str.strip()
    y_lex = y_lex.values
    return X_lex, y_lex
 

def set_up_POS_tag_encoder(
        list_tags
        ) :
    """Sets up a label encoder for the POS tags.

    Args:
        list_tags (list): List of POS tags.

    Returns:
        LabelEncoder: A fitted label encoder for the provided POS tags.
    """
    encoder_tag = LabelEncoder().fit(list_tags)
    return encoder_tag


# vec encoding of words
def alpha_vec2(
        w, 
        mx, 
        max_word_len, 
        dic
        ) :
    """Converts a word to its vector representation using a given dictionary.

    Args:
        w (str): The word to be converted.
        mx (numpy.ndarray): The matrix of word vectors.
        max_word_len (int): The maximum length of words.
        dic (numpy.ndarray): The dictionary of words.

    Returns:
        numpy.ndarray: The vector representation of the word.
    """
    vec = np.zeros((max_word_len, len(dic)))
    for i in range(0, len(w)):
        vec[i] = mx[np.where(dic == w[i])[0][0]]
    vec = vec.astype('float16').flatten()

    vec=vec.astype('float16').flatten()
    vec[vec==np.inf]=0
    vec[vec==-np.inf]=0
    return vec

