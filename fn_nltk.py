from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score

import pandas as pd
import numpy as np

from joblib import dump



def extract_words_and_tags(nested_list):
    # Flatten the nested list of tuples
    words = [word for sentence in nested_list for word, _ in sentence]
    tags = [tag for sentence in nested_list for _, tag in sentence]
    
    # Convert the lists to numpy arrays
    words_array = np.array(words, dtype=object)
    tags_array = np.array(tags, dtype=object)
    
    return words_array, tags_array


# Parse the Kazakh Dependency Treebank in CoNLL-U format
def parse_conllu(file_path):
    sentences = []
    current_sentence = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == '':  # New sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                # CoNLL-U format (columns: ID, FORM, LEMMA, UPOS, XPOS, etc.)
                parts = line.strip().split('\t')
                if len(parts) > 4:
                    word, pos_tag = parts[1], parts[3]  # FORM and UPOS
                    current_sentence.append((word, pos_tag))

    return sentences


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:3],
        'prefix-3': sentence[index][:4],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-3:],
        'suffix-3': sentence[index][-4:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y
