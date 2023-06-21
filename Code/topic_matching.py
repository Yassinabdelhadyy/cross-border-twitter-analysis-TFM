from bertopic import BERTopic
import numpy as np
import pandas as pd
import custom_functions 

f = open("spanish_company/vegan_topic.txt", "r")
vegan_topic = int(f.read())
f.close()


loaded_model = BERTopic.load("spanish_company/model")

# reading the full data data
# sb_path_en = "Sbahn/data/sb_en.csv"
# sb_path_de = "Sbahn/twitter_api/SBahnBerlin_followers_tweets.csv"
# for the sample test 
sb_path_en = "Sbahn/sample/sb_en.csv" 
sb_path_de = "Sbahn/sample/SBahnBerlin_followers_tweets.csv" 


sb_en = custom_functions.df_ready(sb_path_en,sb_path_de)
sb_proc = custom_functions.data_proc(sb_en)



summary_sb = pd.DataFrame({
  "step":["users","tweets"],
  "before":[len(custom_functions.unique(sb_en["author.id"].to_list())),len(custom_functions.unique(sb_en["id"].to_list()))],
  "after":[len(custom_functions.unique(sb_proc["author.id"].to_list())),len(custom_functions.unique(sb_proc["id"].to_list()))]
})
summary_sb["diff_perc"] = np.round(((summary_sb["after"] - summary_sb["before"])/summary_sb["before"])*100,2)

summary_sb

sb_proc["topic_match"] = ""
sb_proc["topic_similarity"] = 0
for index,row in sb_proc.iterrows():
  similar_topics, similarity = loaded_model.find_topics(row["tweet_lemmatized"], top_n=1); # Print results
  sb_proc.at[index, 'topic_match'] = similar_topics[0]
  sb_proc.at[index, 'topic_similarity'] = np.round(similarity[0],2)



sb_proc[["eng","topic_similarity"]][sb_proc["topic_match"] == vegan_topic].sort_values(by="topic_similarity", ascending=False).head()

sb_proc["is_vegan"] = sb_proc['topic_match'] == vegan_topic
## for full data 
# sb_proc.to_csv("SBahn/data/sb_en_topic.csv",index=False)
## for sample
sb_proc.to_csv("SBahn/sample/sb_en_topic.csv",index=False)


all_sb_users = custom_functions.unique(sb_proc["author.id"].to_list())

sb_vegan_users = custom_functions.unique(sb_proc["author.id"][sb_proc["is_vegan"]==True].to_list())

summary_sb_vegan = pd.DataFrame({
  "step":["users","tweets"],
  "all":[len(all_sb_users),len(sb_proc["id"][sb_proc["is_vegan"]==False].to_list())],
  "vegan":[len(sb_vegan_users),len(sb_proc["id"][sb_proc["is_vegan"]==True].to_list())]
})
summary_sb_vegan["perc"] = np.round((summary_sb_vegan["vegan"]/summary_sb_vegan["all"])*100,2)

summary_sb_vegan



# for friends of the target accounts
custom_functions.f_of_f(sb_path_de,sb_proc,0.01)