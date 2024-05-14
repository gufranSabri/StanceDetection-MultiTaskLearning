import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset
from arabert.preprocess import ArabertPreprocessor
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

from model import *
from preprocess import *
from data_generator import *

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

def main(args):
    device = "cpu"
    num_sentiment_labels = 3
    num_sarcasm_labels = 2
    num_stance_labels = 2
    BERT_hidden_state_size = 768
    task_head_layer_size = 256

    pool_arg_index = args.model_path.index("pool")
    end_index = args.model_path[pool_arg_index:].index("_")
    pooling = args.model_path[pool_arg_index+5:pool_arg_index + end_index]

    use_bi_arg_index = args.model_path.index("bi")
    end_index = args.model_path[use_bi_arg_index:].index("_")
    use_bi = args.model_path[use_bi_arg_index+3:use_bi_arg_index + end_index]

    use_gru_arg_index = args.model_path.index("gru")
    end_index = args.model_path[use_gru_arg_index:].index("_")
    use_gru = args.model_path[use_gru_arg_index+4:use_gru_arg_index + end_index]

    bert_arg_index = args.model_path.index("bert")
    end_index = args.model_path[bert_arg_index:].index("_")
    bert_model_indices = args.model_path[bert_arg_index+6:bert_arg_index + end_index]

    ensemble_setting_arg_index = args.model_path.index("ENSEMBLE_")
    end_index = args.model_path[ensemble_setting_arg_index + len("ENSEMBLE_"):].index("_")
    ensemble_setting = args.model_path[ensemble_setting_arg_index:ensemble_setting_arg_index + len("ENSEMBLE_") + end_index]
    ensemble_setting = str(ensemble_settings[ensemble_setting])

    ws = "EQUAL_" if "EQUAL" in args.model_path else ""
    ws = "STATIC_" if "STATIC" in args.model_path else ws
    ws = "RELATIVE_" if "RELATIVE" in args.model_path else ws
    ws = "HEIRARCHICAL_" if "HEIRARCHICAL" in args.model_path else ws
    weighting_setting_arg_index = args.model_path.index(ws)
    end_index = args.model_path[weighting_setting_arg_index + len(ws):].index("_")
    weighting_setting = args.model_path[weighting_setting_arg_index:weighting_setting_arg_index + len(ws) + end_index]
    weighting_setting = str(weighting_settings[weighting_setting])

    bert_models = get_bert_models([int(index) for index in list(bert_model_indices)])
    
    print("MODEL SETTINGS")
    print(f"Model Path: {args.model_path}")
    print(f"Bert Models: {bert_models}")
    print(f"Pooling: {pooling}")
    print(f"Use Bi: {use_bi}")
    print(f"Use GRU: {use_gru}")
    print(f"Ensemble Setting: {ensemble_setting}")
    print(f"Weighting Setting: {weighting_setting}")
    print(f"Save Pred: {args.save_pred}")
    print()

    if "parallel" in args.model_path:
        model = ParrallelMultiTaskModel(
            bert_models,
            ParallelTaskHead(
                num_sentiment_labels, 
                BERT_hidden_state_size,
                task_name="sentiment", 
                pool_bert_output=bool(int(pooling)),
                use_bi=bool(int(use_bi)),
                use_gru=bool(int(use_gru)),
            ),
            ParallelTaskHead(
                num_sarcasm_labels, 
                BERT_hidden_state_size,
                task_name="sarcasm",
                pool_bert_output=bool(int(pooling)),
                use_bi=bool(int(use_bi)),
                use_gru=bool(int(use_gru)),
            ),
            ParallelTaskHead(
                num_stance_labels, 
                BERT_hidden_state_size,
                task_name="stance",
                pool_bert_output=bool(int(pooling)),
                use_bi=bool(int(use_bi)),
                use_gru=bool(int(use_gru)),
            ),
            task_head_layer_size,
            combination_method=int(ensemble_setting),
            weighting_method=int(weighting_setting),
            pool_bert_output=bool(int(pooling)),
            device=device
        ).to(device)
    else:
        model = SequentialMultiTaskModel(
            bert_models,
            SequentialTaskHead(
                num_sentiment_labels,
                BERT_hidden_state_size,
                task_name="sentiment",
                is_first_head = "sentiment" in args.model_path,
                pool_bert_output=bool(int(pooling)),
                use_bi=bool(int(use_bi)),
                use_gru=bool(int(use_gru)),
            ),
            SequentialTaskHead(
                num_sarcasm_labels, 
                BERT_hidden_state_size,
                task_name="sarcasm",
                is_first_head = "sarcasm" in args.model_path,
                pool_bert_output=bool(int(pooling)),
                use_bi=bool(int(use_bi)),
                use_gru=bool(int(use_gru)),
            ),
            SequentialTaskHead(
                num_stance_labels,
                BERT_hidden_state_size,
                task_name="stance",
                pool_bert_output=bool(int(pooling)),
                use_bi=bool(int(use_bi)),
                use_gru=bool(int(use_gru)),
            ),
            task_head_layer_size,
            combination_method=int(ensemble_setting),
            weighting_method=int(weighting_setting),
            pool_bert_output=bool(int(pooling)),
            first_task="sentiment" if "sentiment "in args.model_path else "sarcasm",
            device=device
        ).to(device)

    weights = torch.load(args.model_path, map_location=torch.device(device))
    model.load_state_dict(weights)

    mapping_sarcasm = {"No": 0, "Yes": 1}
    mapping_stance = {"Favor": 1, "Against": 0}
    mapping_sentiment = {"Negative": 0, "Neutral": 1, "Positive": 2}
    reverse_mapping_stance = {1: "Favor", 0: "Against"}
    reverse_mapping_sarcasm = {0: "No", 1: "Yes"}
    reverse_mapping_sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}

    df = pd.read_csv(args.data_path)
    blind_test = "stance" not in df.columns
    if not blind_test:
        df["stance"].replace({"None": np.nan}, inplace=True)
        df = df.dropna(subset=["stance"])
        df = df.sample(frac=1).reset_index(drop=True)
        df = df[["text", "sarcasm", "sentiment", "stance" , "sarcasm:confidence", "sentiment:confidence", "stance:confidence"]]

        df['sarcasm'] = df['sarcasm'].map(lambda x: mapping_sarcasm[x])
        df['sentiment'] = df['sentiment'].map(lambda x: mapping_sentiment[x])
        df['stance'] = df['stance'].map(lambda x: mapping_stance[x])

    arabert_prep = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02-twitter")
    df.text = df.text.apply(lambda x: process_tweet(x))
    df.text = df.text.apply(lambda x: arabert_prep.preprocess(x))

    x_cols = ["text"]
    if not blind_test:
        x_cols = ["text", "sarcasm:confidence", "sentiment:confidence", "stance:confidence"]

    X = df[x_cols]
    y = df[["sarcasm", "sentiment", "stance"]] if not blind_test else None

    encoded_tweets = []
    for bert_model in bert_models:
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        encoded_tweets.append([encode_text(text, tokenizer) for text in X["text"]])

    tweets, dataset, loader = None, None, None

    if not blind_test:
        tweets, sentiments, sarcasms, stances, *confidences = data_generator(encoded_tweets, X[["sarcasm:confidence", "sentiment:confidence", "stance:confidence"]], y)
        dataset = TensorDataset(
            *[torch.cat([item["input_ids"] for item in enc_tweets]) for enc_tweets in tweets],
            *[torch.cat([item["attention_mask"] for item in enc_tweets]) for enc_tweets in tweets],
            torch.tensor(confidences[0]),
            torch.tensor(confidences[1]),
            torch.tensor(confidences[2]),
            torch.tensor(sarcasms),
            torch.tensor(sentiments),
            torch.tensor(stances),
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        tweets, *_ = data_generator(encoded_tweets, None, None)
        dataset = TensorDataset(
            *[torch.cat([item["input_ids"] for item in enc_tweets]) for enc_tweets in tweets],
            *[torch.cat([item["attention_mask"] for item in enc_tweets]) for enc_tweets in tweets],
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)


    preds_stance = []
    truth_stance = []
    for batch in tqdm(loader, desc="Testing"):
        tweet_data, sarc_y, sent_y, stance_y = None, None, None, None
        if blind_test:
            tweet_data = batch
        else:
            *tweet_data, _, _, _, sarc_y, sent_y, stance_y = batch
        
        input_ids, attention_mask = [], []
        for i in range(len(tweet_data)//2):
            input_ids.append(tweet_data[i])
        for i in range(len(tweet_data)//2,len(tweet_data)):
            attention_mask.append(tweet_data[i])

        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i].to(device)
            attention_mask[i] = attention_mask[i].to(device)

        sarcasm_logits, sentiment_logits, stance_logits = model(input_ids, attention_mask)
        # sarcasm_pred = sarcasm_logits.argmax(dim=1).item()
        # sentiment_pred = sentiment_logits.argmax(dim=1).item()

        stance_pred = stance_logits.argmax(dim=1).item()
        
        preds_stance.append(stance_pred)
        if not blind_test:
            truth_stance.append(stance_y.item())

    if not blind_test:
        print(f"Stance Accuracy: {accuracy_score(truth_stance, preds_stance)}")
        print(f"Stance F1: {f1_score(truth_stance, preds_stance, average='macro')}")

    if args.save_pred:
        if not os.path.exists("./res"):
            os.makedirs("./res")

        preds_stance = [reverse_mapping_stance[pred] for pred in preds_stance]
        df["pred"] = preds_stance

        model_name = args.model_path.split("/")[-1].replace(".pt", "")
        pred_file_name = args.data_path.split("/")[-1].replace(".csv", f"_pred_{model_name}.csv")
        pred_file_path = os.path.join("./res", pred_file_name).replace(" ","_")
        df.to_csv(pred_file_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-model_path", dest="model_path", required=True)
    parser.add_argument("-data_path", dest="data_path", required=True)
    parser.add_argument("-save_pred", dest="save_pred", default=False, type=bool)
    args = parser.parse_args()

    models = os.listdir("./models")

    for model in models:
        if ".pt" not in model: continue
        args.model_path = f"./models/{model}"
        main(args)