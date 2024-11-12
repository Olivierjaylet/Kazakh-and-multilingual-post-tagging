# Kazakh-and-multilingual-post-tagging

## Introduction
Part of speach tagging is the process of assigning each word of a sentence a grammatical category, based on its role. Common POS categories include nouns, verbs, adjectives, adverbs, and prepositions.
This stage can be important in natural language processing (NLP) to make the machine understand the role of each word. Being able to predict the POS of each word in a sentence efficiently enough can add important information to many NLP tasks. For example, knowing the POS of a word can help in lemmatization, since in some languages, a word can have different lemmas depending on whether it is a verb or an adjective. In English, the word "saw" can either be from the lemma of the verb "seeing" or of the noun "saw".
In sentiment analysis, identifying the POS tag can also contribute to improve performance, since adverbs and adjectives often refer to feelings or tones. For instance, part-of-speach taggin also aids in other NLP application such as syntactic parsing, machine translation, named entity recognition (NER), and text-to-speech synthesis, where understanding sentence structure improves output accuracy.
However, there are around 7000 languages in the world, and many different roots. Indeed, Latin, Germanic, Turkic, Slavic, Indo-Aryan, or Dravidian languages all have different grammatical structures. The existence of so many languages makes it difficult to generalize classical part-of-speech tag prediction algorithms.

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
