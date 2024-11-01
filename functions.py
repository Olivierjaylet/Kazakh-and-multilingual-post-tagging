from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


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
    """Cleans the input DataFrame by dropping NaN values and removing punctuation.

    Args:
        df (pandas.DataFrame): DataFrame containing 'WORD' and 'POS' columns.

    Returns:
        tuple: A tuple containing:
            - X_lex (numpy.ndarray): Array of cleaned words.
            - Y_lex (numpy.ndarray): Array of cleaned POS tags.
    """

    # Drop rows where 'WORD' is NaN
    df = df.dropna(subset=['WORD'])

    # Remove punctuation characters
    df = df[df['POS'] != 'PUNCT']

    # Some characters to remove
    chars_to_remove = r"[\#\$\%\&\(\)\+\,\-\.\–\’\:\@\']"

    # Removing the characters from the 'WORD' column
    df['WORD'] = df['WORD'].str.replace(chars_to_remove,
                                        '',
                                        regex=True
                                        )
    
    # list of tags we want to predict
    POS_tag_keep = ['NOUN', 'VERB', 'ADJ', 'PROPN', 'NUM', 'ADV', 'ADP', 'CCONJ', 'PRON', 'DET', 'SCONJ', 'AUX', 'INTJ']
    
    df = df[df['POS'].isin(POS_tag_keep)]

    # convert to string type
    df["WORD"] = df["WORD"].astype(str)
    df["POS"] = df["POS"].astype(str)

    df = df.sample(n=10000, # for computational reasons 
                random_state=42
                )

    print("Size dataset : ", df.shape)

    
    X_lex = df['WORD'].str.strip()
    X_lex = X_lex.values

    y_lex = df['POS'].str.strip()
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


def calculate_results(Y_test,
                      Y_train, 
                      predicts_test, 
                      predicts_train
                      ) :
    """Calculates accuracy and F1 score for the test and train datasets.

    Args:
        Y_test (array-like): True labels for the test dataset.
        Y_train (array-like): True labels for the train dataset.
        predicts_test (array-like): Predicted labels for the test dataset.
        predicts_train (array-like): Predicted labels for the train dataset.

    Returns:
        tuple: A tuple containing:
            - test_acc (float): Accuracy of the test dataset.
            - test_f1 (float): F1 score of the test dataset.
            - train_acc (float): Accuracy of the train dataset.
            - train_f1 (float): F1 score of the train dataset.
    """
    test_acc = accuracy_score(Y_test,
                            predicts_test
                            )
    test_f1 = f1_score(Y_test,
                    predicts_test,
                    average = "weighted"
                    )

    train_acc = accuracy_score(Y_train,
                            predicts_train
                            )
    train_f1 = f1_score(Y_train,
                        predicts_train,
                        average = "weighted"
                        )
    
    return test_acc, test_f1, train_acc, train_f1


def per_tag_accuracy(
        y_true, 
        y_pred, 
        list_tags, 
        encoder
        ) :
    """Calculates accuracy per POS tag.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        list_tags (list): List of POS tags.
        encoder (LabelEncoder): Label encoder for transforming tags.

    Returns:
        pandas.DataFrame: DataFrame containing POS tags and their respective accuracies.
    """

    tag_names = []
    accuracies = []

    for tag in list_tags:
        encoded_tag = encoder.transform([tag])[0]
        idx = np.where(y_true == encoded_tag)
        acc = accuracy_score(y_true[idx], 
                             y_pred[idx]
                             )
        tag_names.append(tag)
        accuracies.append(acc)

    df_accuracy = pd.DataFrame({
        'Tag': tag_names,
        'Accuracy': accuracies
    })

    return df_accuracy


def mistake_frequency_by_word_type(
        y_true, 
        y_pred, 
        list_tags, 
        encoder
        ) :
    """Calculates the frequency of mistakes by word type in the predictions.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        list_tags (list): List of POS tags.
        encoder (LabelEncoder): Label encoder for transforming tags.

    Returns:
        pandas.DataFrame: DataFrame of mistake frequencies sorted by frequency.
    """

    y_true_decoded = encoder.inverse_transform(y_true)
    y_pred_decoded = encoder.inverse_transform(y_pred)

    cm = confusion_matrix(y_true_decoded, 
                          y_pred_decoded, 
                          labels=list_tags
                          )

    cm_df = pd.DataFrame(cm, 
                         index=list_tags, 
                         columns=list_tags
                         )

    mistake_freq_records = []

    for true_tag in list_tags:
        for pred_tag in list_tags:
            if true_tag != pred_tag:
                frequency = cm_df.loc[true_tag, pred_tag]
                if frequency > 0:
                    mistake_freq_records.append({'From Tag': true_tag, 
                                                 'To Tag': pred_tag, 
                                                 'Frequency': frequency}
                                                 )

    mistake_freq_df = pd.DataFrame(mistake_freq_records)

    return mistake_freq_df.sort_values(by='Frequency', ascending=False)


def tag_prediction_nb(
        y_true, 
        y_pred, 
        list_tags, 
        encoder
        ) :
    """Calculates the number of correct and incorrect predictions per tag.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        list_tags (list): List of POS tags.
        encoder (LabelEncoder): Label encoder for transforming tags.

    Returns:
        pandas.DataFrame: DataFrame containing POS tags and counts of correct/incorrect predictions.
    """

    tag_names = []
    correct_counts = []
    incorrect_counts = []

    for tag in list_tags:
        encoded_tag = encoder.transform([tag])[0]
        idx = np.where(y_true == encoded_tag)
        
        correct = np.sum(y_true[idx] == y_pred[idx])
        incorrect = len(y_true[idx]) - correct

        tag_names.append(tag)
        correct_counts.append(correct)
        incorrect_counts.append(incorrect)

    df_distribution = pd.DataFrame({
        'Tag': tag_names,
        'Correct Predictions': correct_counts,
        'Incorrect Predictions': incorrect_counts
    })

    return df_distribution



########################################################################################
####################################### GRAPHICS #######################################
########################################################################################

def plot_confusion_matrix(
        Y, 
        predicts, 
        list_tags, 
        title, 
        lang
        ) :
    # Compute confusion matrix
    cm = confusion_matrix(Y, 
                          predicts, 
                          labels=np.arange(len(list_tags)))

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=list_tags,
                yticklabels=list_tags,
                ax=ax)

    plot_title = 'Confusion Matrix of ' + title + " for " + lang + "corpus"
    ax.set_title(plot_title)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    plt.tight_layout()

    return fig



def plot_dist_predictions(
        df_tag_dist, 
        lang
        ) :
    # Set the title for the plot
    title = "Distribution of Correct and Incorrect Predictions for " + lang + ' corpus'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_tag_dist.plot(kind='bar', 
                     x='Tag', 
                     stacked=True, 
                     ax=ax) 
    
    ax.set_title(title)

    return fig


def save_graph_to_folder(
        fig, 
        lang, 
        filename
        ) :

    # Save the graph to the folder
    file_path = os.path.join('graphs', lang, filename)
    fig.savefig(file_path)
    plt.close(fig)  # Close the figure after saving