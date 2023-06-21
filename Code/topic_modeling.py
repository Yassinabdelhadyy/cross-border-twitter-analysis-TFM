from bertopic import BERTopic
import numpy as np
import pandas as pd
# from nltk.corpus import stopwords
# import nltk, re, random, os
# nltk.download('wordnet')
import custom_functions 
import os


# reading the full data data
# h_path_en = "spanish_company/data/h_en.csv"
# h_path_es = "spanish_company/twitter_api/spanish_company_followers_tweets.csv"
# for the sample test 
h_path_en ="spanish_company/sample/h_en.csv" 
h_path_es ="spanish_company/sample/spanish_company_followers_tweets.csv" 

# adding authur id
h_en = custom_functions.df_ready(h_path_en,h_path_es)

# data processing
h_proc = custom_functions.data_proc(h_en)

summary_h = pd.DataFrame({
  "step":["users","tweets"],
  "before":[len(custom_functions.unique(h_en["author.id"].to_list())),len(custom_functions.unique(h_en["id"].to_list()))],
  "after":[len(custom_functions.unique(h_proc["author.id"].to_list())),len(custom_functions.unique(h_proc["id"].to_list()))]
})
summary_h["diff_perc"] = np.round(((summary_h["after"] - summary_h["before"])/summary_h["before"])*100,2)



# list of the tweets
data_list = h_proc["tweet_lemmatized"].to_list()

topic_model = BERTopic()
# applying the model on the list of tweets
topics, probs = topic_model.fit_transform(data_list)

topic_labels = topic_model.generate_topic_labels(nr_words = 3,topic_prefix = False,word_length = 15,separator = " - ")
topic_model.set_topic_labels(topic_labels)

topicsa = topic_model.get_topic_info()
topicsa['is_vegan'] = topicsa['Name'].str.contains('vegan')
topicsa
# topicsa.to_csv("eng/topics.csv",index=False)

vegan_topic = topicsa["Topic"][topicsa["is_vegan"]==True].to_list()[0]


# Get the topic predictions
topic_prediction = topic_model.topics_[:]# Save the predictions in the dataframe
h_proc['topic_prediction'] = topic_prediction

h_proc["is_vegan"] = h_proc['topic_prediction'] == vegan_topic
h_proc.to_csv("spanish_company/sample/h_en_topic.csv",index=False)

all_h_users = custom_functions.unique(h_proc["author.id"].to_list())
h_vegan_users = custom_functions.unique(h_proc["author.id"][h_proc["is_vegan"]==True].to_list())


summary_h_vegan = pd.DataFrame({
  "step":["users","tweets"],
  "all":[len(all_h_users),len(h_proc["id"][h_proc["is_vegan"]==False].to_list())],
  "vegan":[len(h_vegan_users),len(h_proc["id"][h_proc["is_vegan"]==True].to_list())]
})
summary_h_vegan["perc"] = np.round((summary_h_vegan["vegan"]/summary_h_vegan["all"])*100,2)

print(summary_h_vegan)




if sum([item=="model" for item in os.listdir("spanish_company")])<1:
    topic_model.save("model", serialization="pytorch", save_ctfidf=True) 



####################################
# Visualize top topic keywords
if sum([item=="key_words.html" for item in os.listdir("spanish_company/Plots")])<1:
  key_words = topic_model.visualize_barchart(top_n_topics=12)
  key_words.write_html("spanish_company/Plots/key_words.html")

# # Visualize term rank decrease
if sum([item=="term_rank.html" for item in os.listdir("spanish_company/Plots")])<1:
  term_rank = topic_model.visualize_term_rank(top_n_topics=12)
  term_rank.write_html("spanish_company/Plots/term_rank.html")

# # Visualize intertopic distance
if sum([item=="intertopic_distance.html" for item in os.listdir("spanish_company/Plots")])<1:
  intertopic_distance = topic_model.visualize_topics(top_n_topics=12)
  intertopic_distance.write_html("spanish_company/Plots/intertopic_distance.html")


# # Visualize connections between topics using hierachical clustering
if sum([item=="conn_bet_topics.html" for item in os.listdir("spanish_company/Plots")])<1:
  conn_bet_topics = topic_model.visualize_hierarchy(top_n_topics=10)
  conn_bet_topics.write_html("spanish_company/Plots/conn_bet_topics.html")

# # Visualize similarity using heatmap
if sum([item=="heatmap.html" for item in os.listdir("spanish_company/Plots")])<1:
  heatmap = topic_model.visualize_heatmap(top_n_topics=12)
  heatmap.write_html("spanish_company/Plots/heatmap.html")

# saving the desired topic into text file to further use it
if sum([item=="vegan_topic.txt" for item in os.listdir("spanish_company")])<1:
    write_topic = open("spanish_company/vegan_topic.txt", "x")
    write_topic.write(str(vegan_topic))
    write_topic.close()