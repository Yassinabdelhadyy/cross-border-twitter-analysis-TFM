import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')

# stop words
stop = stopwords.words('english')
# adding words to the stop words
custom_words = ["the","and"]
# merging the 2 lists
stop += custom_words
wn = nltk.WordNetLemmatizer()

def data_proc(df):
   #changing the column type of the tweet to string 
  df["eng"]=df["eng"].astype(str)
  # creating a new column 
  df["word_count"] = 0
  for index,row in df.iterrows():
    wordss = len(row["eng"].split())
    df.at[index, 'word_count'] = wordss
  # filtering rows with higher than 3 words
  df = df[df['word_count'] > 3] 
  # lower the the tweets to be able to remove the stop words later
  df['eng'] =df['eng'].str.lower()
  # removing the stop words
  df['tweet_without_stopwords'] = df['eng'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  # Lemmatization  https://medium.com/grabngoinfo/topic-modeling-with-deep-learning-using-python-bertopic-cf91f5676504
  df['tweet_lemmatized'] = df['tweet_without_stopwords'].apply(lambda x: ' '.join([wn.lemmatize(w) for w in x.split() ]))
  return(df)

def unique(list_1):
  return(list(dict.fromkeys(list_1)))


def df_ready(path_en,path_es):
  df_en = pd.read_csv(path_en)
  df_es = pd.read_csv(path_es)[["id","author.id"]]
  df_en = df_en.merge(df_es, on='id', how='left')
  return(df_en)

def f_of_f(tweets_file_path,target_account_df,perc):
  test = target_account_df[["id"]][target_account_df["is_vegan"]==True]
  all_es = pd.read_csv(tweets_file_path)[["id","author.id","author.public_metrics.followers_count"]]
  df_usr_cnt = test.merge(all_es, on='id', how='left')
  df_usr_cnt = df_usr_cnt.drop(['id'], axis=1)
  df_usr_cnt = df_usr_cnt.drop_duplicates()
  number_of_extra_people = np.round(df_usr_cnt["author.public_metrics.followers_count"].sum()*perc)
  return(number_of_extra_people)
