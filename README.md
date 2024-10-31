# Kazakh-and-multilingual-post-tagging

## Project Overview
This project focuses on adapting a lemmatizer based on a Random Forest classifier to predict Part-of-Speech (POS) tagging for Kazakh words. 
The objective was to create an efficient model that can accurately identify the grammatical category of Kazakh words, which is crucial for various natural language processing tasks.
I then compared the results with a corpus of turkish token (language with same roots), as well as a corpus with english tokens.

## Content
The notebook main.ipynb displays some results table for the three languages.
While running it, some graphics are also saved in the folder graphs.
Finally, to make the notebook clean and readable, every hand-made functions were saved and comented in function.py.

## Analysis & conclusion (TO DO)

## Data Source
Source of the 3 datasets :
Kazakh corpus : https://github.com/nlacslab/kazdet/blob/master/data/kdt-NLANU-0.01.connlu.txt.7z. 
English corpus : https://github.com/UniversalDependencies/UD_English-EWT/blob/master/en_ewt-ud-dev.conllu
Turkish corpus : https://github.com/UniversalDependencies/UD_Turkish-Kenet/blob/master/tr_kenet-ud-dev.conllu

## Methodology
The algorithm employed for this project is the Extra Trees classifier.
Scikit-learn library was used to run this model : https://scikit-learn.org/dev/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
Original paper of the methodology adapted :  https://www.scielo.org.mx/scielo.php?pid=S1405-55462020000301353&script=sci_arttext&tlng=en
