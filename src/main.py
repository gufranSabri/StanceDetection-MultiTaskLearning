import os
import pandas as pd
import numpy as np
import random
import re
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from arabert.preprocess import ArabertPreprocessor
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from model import *
from preprocess import *
from data_generator import *
from lr_scheduler import *
from training_loop import *

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_bert_models(indices):
    bert_models = [
        "aubmindlab/bert-base-arabertv02-twitter", 
        "aubmindlab/bert-base-arabertv02",
        "UBC-NLP/MARBERT",
        "CAMeL-Lab/bert-base-arabic-camelbert-da"
    ]
    return [bert_models[i] for i in indices]

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

def main(args):
    seed = 42

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.device == 'mps':
        torch.mps.manual_seed(seed)
        torch.backends.mps.deterministic=True
        torch.backends.mps.benchmark = False
    elif args.device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    #CONFIG=======================================================================================
    task_head_layer_size = 256
    BERT_hidden_state_size = 768
    batch_size = 4
    num_epochs = 1
    learning_rate = 2e-5
    weight_decay = 1e-5
    dropout = 0.1
    #CONFIG=======================================================================================

    # Load Data
    df = pd.read_csv("./data/Mawqif_AllTargets_Train.csv")
    df["stance"].replace({"None": np.nan}, inplace=True)
    df = df.dropna(subset=["stance"])
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[["text", "sarcasm", "sentiment", "stance" , "sarcasm:confidence", "sentiment:confidence", "stance:confidence"]]

    # Preprocess Data
    mapping_sarcasm = {"No": 0, "Yes": 1}
    mapping_stance = {"Favor": 1, "Against": 0}
    mapping_sentiment = {"Negative": 0, "Neutral": 1, "Positive": 2}

    df['sarcasm'] = df['sarcasm'].map(lambda x: mapping_sarcasm[x])
    df['sentiment'] = df['sentiment'].map(lambda x: mapping_sentiment[x])
    df['stance'] = df['stance'].map(lambda x: mapping_stance[x])

    arabert_prep = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02-twitter")
    df.text = df.text.apply(lambda x: process_tweet(x))
    df.text = df.text.apply(lambda x: arabert_prep.preprocess(x))

    X = df[["text", "sarcasm:confidence", "sentiment:confidence", "stance:confidence"]]
    y = df[["sarcasm", "sentiment", "stance"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape, "X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

    encoded_tweets_train = []
    encoded_tweets_test = []
    bert_models = get_bert_models([int(index) for index in list(args.bert_models)])
    for bert_model in bert_models:
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        encoded_tweets_train.append([encode_text(text, tokenizer) for text in X_train["text"]])
        encoded_tweets_test.append([encode_text(text, tokenizer) for text in X_test["text"]])

    tweets, sentiments, sarcasms, stances, *confidences = data_generator(encoded_tweets_train, X_train[["sarcasm:confidence", "sentiment:confidence", "stance:confidence"]], y_train)
    train_dataset = TensorDataset(
        *[torch.cat([item["input_ids"] for item in enc_tweets]) for enc_tweets in tweets],
        *[torch.cat([item["attention_mask"] for item in enc_tweets]) for enc_tweets in tweets],
        torch.tensor(confidences[0]),
        torch.tensor(confidences[1]),
        torch.tensor(confidences[2]),
        torch.tensor(sarcasms),
        torch.tensor(sentiments),
        torch.tensor(stances),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    tweets, sentiments, sarcasms, stances, *confidences = data_generator(encoded_tweets_test, X_test[["sarcasm:confidence", "sentiment:confidence", "stance:confidence"]], y_test)
    val_dataset = TensorDataset(
        *[torch.cat([item["input_ids"] for item in enc_tweets]) for enc_tweets in tweets],
        *[torch.cat([item["attention_mask"] for item in enc_tweets]) for enc_tweets in tweets],
        torch.tensor(confidences[0]),
        torch.tensor(confidences[1]),
        torch.tensor(confidences[2]),
        torch.tensor(sarcasms),
        torch.tensor(sentiments),
        torch.tensor(stances),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    num_sarcasm_labels = len(df.sarcasm.unique())
    num_sentiment_labels = len(df.sentiment.unique())
    num_stance_labels = len(df.stance.unique())

    num_sarcasm_labels, num_sentiment_labels, num_stance_labels

    model = None
    print("\nINITIALIZING MODEL")
    if args.model == 'parallel':
        model = ParrallelMultiTaskModel(
            bert_models,
            ParallelTaskHead(
                num_sentiment_labels, 
                BERT_hidden_state_size,
                task_name="sentiment", 
                pool_bert_output=bool(int(args.pooling)),
                use_bi=bool(int(args.use_bi)),
                use_gru=bool(int(args.use_gru)),
            ),
            ParallelTaskHead(
                num_sarcasm_labels, 
                BERT_hidden_state_size,
                task_name="sarcasm",
                pool_bert_output=bool(int(args.pooling)),
                use_bi=bool(int(args.use_bi)),
                use_gru=bool(int(args.use_gru)),
            ),
            ParallelTaskHead(
                num_stance_labels, 
                BERT_hidden_state_size,
                task_name="stance",
                pool_bert_output=bool(int(args.pooling)),
                use_bi=bool(int(args.use_bi)),
                use_gru=bool(int(args.use_gru)),
            ),
            task_head_layer_size,
            combination_method=int(args.ensemble_setting),
            weighting_method=int(args.weighting_setting),
            pool_bert_output=bool(int(args.pooling)),
            device=args.device
        ).to(args.device)
    else:
        model = SequentialMultiTaskModel(
            bert_models,
            SequentialTaskHead(
                num_sentiment_labels if args.first_task == "sentiment" else num_sarcasm_labels, 
                BERT_hidden_state_size,
                task_name=args.first_task,
                is_first_head=True,
                pool_bert_output=bool(int(args.pooling)),
                use_bi=bool(int(args.use_bi)),
                use_gru=bool(int(args.use_gru)),
            ),
            SequentialTaskHead(
                num_sentiment_labels if args.first_task == "sarcasm" else num_sarcasm_labels, 
                BERT_hidden_state_size,
                task_name="sarcasm" if args.first_task == "sentiment" else "sentiment",
                pool_bert_output=bool(int(args.pooling)),
                use_bi=bool(int(args.use_bi)),
                use_gru=bool(int(args.use_gru)),
            ),
            SequentialTaskHead(
                num_stance_labels,
                BERT_hidden_state_size,
                task_name="stance",
                pool_bert_output=bool(int(args.pooling)),
                use_bi=bool(int(args.use_bi)),
                use_gru=bool(int(args.use_gru)),
            ),
            task_head_layer_size,
            combination_method=int(args.ensemble_setting),
            weighting_method=int(args.weighting_setting),
            pool_bert_output=bool(int(args.pooling)),
            device=args.device
        ).to(args.device)
    print("------------------------------------\n")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    ce_loss = nn.CrossEntropyLoss()
    lr_scheduler=LinearDecayLR(optimizer, num_epochs, int(num_epochs*0.5))

    print("SETTINGS")
    print("Model:", args.model)
    print("BERT Models:", bert_models)
    print("Task Head Layer Size:", task_head_layer_size)
    print("Batch Size:", batch_size)
    print("Num Epochs:", num_epochs)
    print("Learning Rate:", learning_rate)
    print("Weight Decay:", weight_decay)
    print("Dropout:", dropout)
    print("Device:", args.device)
    print("Seed:", seed)
    print("Ensemble Method:", list(ensemble_settings.keys())[list(ensemble_settings.values()).index(int(args.ensemble_setting))])
    print("Weighting Method:", list(weighting_settings.keys())[list(weighting_settings.values()).index(int(args.weighting_setting))])
    print("Pool BERT Output:", bool(int(args.pooling)))
    print("Use Bi:", bool(int(args.use_bi)))
    print("Use GRU:", bool(int(args.use_bi)))
    print("------------------------------------\n")

    best_valid_stance_acc = None
    if args.model == 'parallel':
        print("TRAINING PARALLEL MTL MODEL")
        model, best_valid_stance_acc = parallel_trainer(
            train_loader, 
            val_loader, 
            model, 
            optimizer, 
            lr_scheduler, 
            ce_loss,
            num_epochs, 
            WEIGHTING_METHOD=int(args.weighting_setting), 
            device = args.device
        )
    else:
        print("TRAINING SEQUENTIAL MTL MODEL")
        model, best_valid_stance_acc = sequential_trainer(
            train_loader, 
            val_loader, 
            model, 
            optimizer, 
            lr_scheduler, 
            ce_loss,
            num_epochs, 
            WEIGHTING_METHOD=int(args.weighting_setting), 
            device = args.device
        )
    
    ensm = list(ensemble_settings.keys())[list(ensemble_settings.values()).index(int(args.ensemble_setting))]
    wsm = list(weighting_settings.keys())[list(weighting_settings.values()).index(int(args.weighting_setting))]
    torch.save(model.state_dict(), f"./models/{args.model}_fs.{args.first_task}_{ensm}_{wsm}_berts.{args.bert_models}_pool.{args.pooling}_bi.{args.use_bi}_gru.{args.use_gru}_stanceAcc{best_valid_stance_acc:.2f}.pt")
        
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-model',dest='model', default='parallel')
    parser.add_argument('-first',dest='first_task', default='sentiment')
    parser.add_argument('-es',dest='ensemble_setting', default='0')
    parser.add_argument('-ws',dest='weighting_setting', default='0')
    parser.add_argument('-bert',dest='bert_models', default='0')
    parser.add_argument('-pool',dest='pooling', default='0')
    parser.add_argument('-bi',dest='use_bi', default='0')
    parser.add_argument('-gru',dest='use_gru', default='0')
    parser.add_argument('-device',dest='device', default='cuda')
    args=parser.parse_args()

    if not os.path.exists("./models"):
        os.makedirs("./models")

    main(args)