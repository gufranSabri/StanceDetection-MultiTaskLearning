import os
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def data_generator(encoded_tweets, confidences, labels):    
    tweets, sentiments, sarcasms, stances = [], [], [], []
    sarcasm_confidence, sentiment_confidence, stance_confidence = [], [], []
    
    for i in range(len(encoded_tweets)):
        model_specific_tweets = []
        for j in range(len(encoded_tweets[i])):
            model_specific_tweets.append(encoded_tweets[i][j])            
        tweets.append(model_specific_tweets)
        
    for i in range(len(labels)):
        sentiments.append(labels.sentiment.iloc[i])
        sarcasms.append(labels.sarcasm.iloc[i])
        stances.append(labels.stance.iloc[i])
        sarcasm_confidence.append(confidences["sarcasm:confidence"].iloc[i])
        sentiment_confidence.append(confidences["sentiment:confidence"].iloc[i])
        stance_confidence.append(confidences["stance:confidence"].iloc[i])
            
    return tweets, sentiments, sarcasms, stances, sentiment_confidence, sarcasm_confidence, stance_confidence