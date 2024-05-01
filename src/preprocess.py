import os
import re
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def encode_text(text, tokenizer):
    return tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

def remove_hash_URL_MEN(text):
    text = re.sub(r'#',' ',text)
    text = re.sub(r'_',' ',text)
    text = re.sub(r'URL','',text)
    text = re.sub(r'MENTION','',text)
    return text

def normalize_arabic(text):
    text = re.sub("[إآ]", "ا", text)
    text = re.sub("گ", "ك", text)
    return text

def process_tweet(tweet):     
    tweet=remove_hash_URL_MEN(tweet)
    tweet = re.sub('@[^\s]+', ' ', str(tweet))
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',str(tweet))    
    tweet= normalize_arabic(str(tweet))
    
    return tweet