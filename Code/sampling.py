import random
import pandas as pd
from cryptography.fernet import Fernet

f = open("hashing/key.txt","r")
key = bytes(f.readlines()[0],"utf-8")
f.close()
fernet = Fernet(key)
def rnd_sampling(tweets_path,number_rows):
    df = pd.read_csv(tweets_path)   
    random_list = random.sample(range(0,len(df)),number_rows)
    df = df.iloc[random_list] 
    
    df = df[["id","text","author.id","author.public_metrics.followers_count"]]

    for index,row in df.iterrows():
        try:
            fake_tweet_id = fernet.encrypt(str(row["id"]).encode())
            df.at[index, 'id'] = fake_tweet_id
        except:
            pass
        try:
            fake_user_id = fernet.encrypt(str(row["author.id"]).encode())
            df.at[index, 'author.id'] = fake_user_id
        except:
            pass
        df.to_csv(tweets_path,index=False)
    return(print("done"))



h_tweets = "spanish_company/twitter_api/spanish_company_followers_tweets.csv"
sb_tweets = "SBahn/twitter_api/SBahnBerlin_followers_tweets.csv"



rnd_sampling(h_tweets,1000).to_csv("spanish_company/sample/spanish_company_followers_tweets.csv", index = False)
rnd_sampling(sb_tweets,1000).to_csv("SBahn/sample/SBahnBerlin_followers_tweets.csv", index = False)
