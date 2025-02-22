from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score


import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

import os




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
    
    test_recall = recall_score(Y_test,
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
    train_recall = recall_score(Y_train,
                        predicts_train,
                        average = "weighted"
                        )
    
    
    return test_acc, test_f1, test_recall, train_acc, train_f1, train_recall


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
        encoder,
        top_n=10
        ):
    """
    Calculates and plots the normalized frequency of mistakes by word type.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        list_tags (list): List of POS tags.
        encoder (LabelEncoder): Label encoder for transforming tags.
        top_n (int): Number of top frequent mistakes to plot.

    Returns:
        pandas.DataFrame: DataFrame of normalized mistake frequencies sorted by frequency.
    """
    # Decode labels
    y_true_decoded = encoder.inverse_transform(y_true)
    y_pred_decoded = encoder.inverse_transform(y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_true_decoded, 
                          y_pred_decoded, 
                          labels=list_tags)

    # Create confusion matrix DataFrame
    cm_df = pd.DataFrame(cm, 
                         index=list_tags, 
                         columns=list_tags)

    mistake_freq_records = []

    # Calculate mistake frequencies
    for true_tag in list_tags:
        for pred_tag in list_tags:
            if true_tag != pred_tag:
                frequency = cm_df.loc[true_tag, pred_tag]
                if frequency > 0:
                    # Normalize frequency by the total occurrences of the true tag
                    total_true_tag = cm_df.loc[true_tag].sum()
                    normalized_frequency = frequency / total_true_tag
                    mistake_freq_records.append({'From Tag': true_tag, 
                                                 'To Tag': pred_tag, 
                                                 'Frequency': normalized_frequency,
                                                 'nb_mispredictions' : frequency})

    # Create DataFrame of mistakes
    mistake_freq_df = pd.DataFrame(mistake_freq_records)

    # Sort by normalized frequency
    return  mistake_freq_df.sort_values(by='nb_mispredictions', ascending=False)


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

    plot_title = 'Confusion Matrix of ' + title + " for " + lang + " corpus"
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
