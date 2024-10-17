# Adapt a lemmatize for POS tagging of Kazakh language

## Project Overview
This project focuses on adapting a lemmatizer based on a Random Forest classifier to predict Part-of-Speech (POS) tagging for Kazakh words. The objective was to create an efficient model that can accurately identify the grammatical category of Kazakh words, which is crucial for various natural language processing tasks.

## Data Source
The dataset used in this project was sourced from https://github.com/nlacslab/kazdet/blob/master/data/kdt-NLANU-0.01.connlu.txt.7z. 
It contains annotated samples of Kazakh text, providing more than 700K tokens to split for training and testing the model.

## Methodology
The algorithm employed for this project is the Extra Trees classifier, which is known for its robustness and effectiveness in classification tasks. 
Scikit-learn library was used to run this model : https://scikit-learn.org/dev/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
Original paper of the methodology adapted :  https://www.scielo.org.mx/scielo.php?pid=S1405-55462020000301353&script=sci_arttext&tlng=en
