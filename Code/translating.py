from transformers import MarianMTModel, MarianTokenizer
import pandas as pd


def translate_full(orignal_file_path,target_file_path,lang):
    for item in range(len(orignal_file_path)):
        data = pd.read_csv(orignal_file_path[item])
        data["eng"] = ""
        # Load the pre-trained model and tokenizer
        model_name = lang[item]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        counter = 0
        for index,row in data.iterrows():
            counter += 1
            # break
            try:
                inputs = tokenizer(row["clean_text"], return_tensors="pt")
                outputs = model.generate(**inputs)
                english_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                data.at[index, 'eng'] = english_text
                # saving after 50 itterations 
                if counter == 50:
                    print(index)
                    # break
                    data.to_csv(target_file_path[item],index=False)      
                    counter = 0          
            except:
                pass
            
        # saving the full file
        data.to_csv(target_file_path[item],index=False)


languages = ["Helsinki-NLP/opus-mt-es-en","Helsinki-NLP/opus-mt-de-en"]


# all data
# original_files = [ "spanish_company/twitter_api/spanish_company_followers_tweets.csv","SBahn/twitter_api/SBahnBerlin_followers_tweets.csv"]
# saving_files = ["spanish_company/data/h_en.csv","SBahn/data/sb_en.csv"]
# translate_full(spanish_company_full,other_full)

#sample
original_files = ["spanish_company/sample/h_es.csv","SBahn/sample/sb_de.csv"]
saving_files = ["spanish_company/sample/h_en.csv","SBahn/sample/sb_en.csv"]

languages = ["Helsinki-NLP/opus-mt-de-en"]
original_files = ["SBahn/sample/sb_de.csv"]
saving_files = ["SBahn/sample/sb_en.csv"]
translate_full(original_files,saving_files,languages)