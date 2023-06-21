## How to run the scripts

pip install torch
pip install tensorflow
pip install transformers
pip install SentencePiece 
pip install sacremoses
pip install pandas
pip install numpy
pip install bertopic


- packages Needed:

- torch

- tensorflow

- transformers

- SentencePiece

- sacremoses

- pandas

- numpy

- re 

- string

- from bertopic import BERTopic 

- numpy

- os

- from nltk.corpus import stopwords

- nltk

- nltk.download('wordnet')


- custom_functions

    **which are custom functions created in custom_functions.py**

<hr>

- sampling.py **DO NOT RUN**

    Generating a sample from all the tweets 

- twitter_clean.py  **First to run**

    Cleaning the data set from all the unnecessary  text from the tweets

- translating.py    **Second to run**

    Translating the clean text to english

- topic_modeling.py **DO NOT RUN ON THE SAMPLE -- Third to run**

    Generating the topic modelling algorithm

    **A model has been generated on the full data in spainsh_company/model**

- topic_matching.py **Fourth to run**

    Matching the tweets from SBahnBerlin sample to the topics 