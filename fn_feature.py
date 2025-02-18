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


def get_values(sentences):
    X_lex = np.array([word for sentence in sentences for word, _ in sentence], dtype=object)
    y_lex = np.array([pos for sentence in sentences for _, pos in sentence], dtype=object)
    return X_lex, y_lex



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

