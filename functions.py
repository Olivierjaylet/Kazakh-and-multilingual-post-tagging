from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np



# functions
def get_values(df):
    X_lex = df['WORD'].str.strip()
    X_lex = X_lex.values

    y_lex = df['POS'].str.strip()
    y_lex = y_lex.values

    return X_lex, y_lex
def list_all_POS_tags(y) :
    list_tags = []
    for tag_list in y :
        tags = tag_list.split()
        for tag in tags :
            if tag not in list_tags :
                list_tags.append(tag)
    return list_tags

def set_up_POS_tag_encoder(list_tags) :
    encoder_tag = LabelEncoder().fit(list_tags)
    return encoder_tag

# vec encoding of words
def alpha_vec2(w, mx, max_word_len, dic):
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
                      ):

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

def plot_confusion_matrix(Y, predicts, list_tags, title):
  
  cm = confusion_matrix(Y, 
                        predicts, 
                        labels = np.arange(len(list_tags))
                        )

  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  plt.figure(figsize=(8, 6))
  sns.heatmap(cm_normalized,
              annot=True,
              fmt='.2f',
              cmap='Blues',
              xticklabels=list_tags,
              yticklabels=list_tags)
  title = 'Confusion Matrix of ' + title
  plt.title(title)
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')

  plt.tight_layout()
  plt.show()


def per_tag_accuracy(y_true, y_pred, list_tags, encoder):
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


def tag_prediction_nb(y_true, y_pred, list_tags, encoder):

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

def mistake_frequency_by_word_type(y_true, y_pred, list_tags, encoder):
    
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


def clean_data(df):
    # Drop rows where 'WORD' is NaN
    df = df.dropna(subset=['WORD'])

    # Remove punctuation characters
    df = df[df['POS'] != 'PUNCT']

    # Some characters to remove
    chars_to_remove = r"[\#\$\%\&\(\)\+\,\-\.\–\’\:\@]"

    # Removing the characters from the 'WORD' column
    df['WORD'] = df['WORD'].str.replace(chars_to_remove,
                                        '',
                                        regex=True
                                        )
    df = df[df['POS'] != 'X']

    df["WORD"] = df["WORD"].astype(str)
    df["POS"] = df["POS"].astype(str)

    df = df.sample(n=2000, # for computational reasons 
                random_state=42
                )

    # Split into train & test sets
    X_lex, Y_lex = get_values(df)

    return X_lex, Y_lex
