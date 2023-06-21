import pandas as pd
import re, string

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def tweet_clean(df):
    df["clean_text"] = ""
    regexp_img = "https?://pbs.twimg.com/[^ ]+[.](jpg|png)"
    regexp_link = "https?://[^ ]+"
    regexp_mentions = "@\\w+"
    regex_linebreak = "\\\\n"
    regex_double_linebreak = "\\\\n\\\\n"
    regex_rec_spaces = "\\s+"
    for index,row in df.iterrows():    
        clean = re.sub(regexp_img, "", row["text"])
        clean = re.sub(regexp_link, "", clean)
        clean = re.sub(regexp_mentions, "", clean)
        clean = re.sub(regex_linebreak, "", clean)
        clean = re.sub(regex_double_linebreak, "", clean)
        clean = clean.translate(str.maketrans('', '', string.punctuation))
        clean = re.sub("[0-9]", "", clean)
        clean = remove_emoji(clean)
        clean = re.sub(regex_rec_spaces, " ", clean)
        clean = clean.strip()
        clean = clean.lower()
        df.at[index, 'clean_text'] = clean





# spanish_company Tweets
# spanish_company_es = pd.read_csv("spanish_company/twitter_api/spanish_company_followers_tweets.csv")
spanish_company_es = pd.read_csv("spanish_company/sample/spanish_company_followers_tweets.csv")
spanish_company_es_sub = spanish_company_es[["id", "text"]]
spanish_company_es_sub = spanish_company_es_sub[~spanish_company_es_sub.text.str.startswith("RT @")]


tweet_clean(spanish_company_es_sub)
spanish_company_es_sub.to_csv("spanish_company/sample/h_es.csv",index=False)


# Sbahn Berlin Tweets
# sbahn_tweets_es = pd.read_csv("SBahn/twitter_api/SBahnBerlin_followers_tweets.csv")
sbahn_tweets_de = pd.read_csv("SBahn/sample/SBahnBerlin_followers_tweets.csv")
sbahn_tweets_de_sub = sbahn_tweets_de[["id", "text"]]
sbahn_tweets_de_sub = sbahn_tweets_de_sub[~sbahn_tweets_de_sub.text.str.startswith("RT @")]
tweet_clean(sbahn_tweets_de_sub)
sbahn_tweets_de_sub.to_csv("SBahn/sample/sb_de.csv",index=False)
