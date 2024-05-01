import os
from tqdm import tqdm
import os
import torch
import warnings
from model import weighting_settings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def sequential_trainer(
        train_loader, 
        val_loader, 
        model, 
        optimizer, 
        lr_scheduler,
        ce_loss, 
        num_epochs=20,
        first_task="sentiment", 
        WEIGHTING_METHOD = weighting_settings["EQUAL_WEIGHTING"] ,
        patience=5,
        device="cuda",
    ):

    best_valid_accs = {"stance": -float('inf'), "sarcasm": -float('inf'), "sentiment": -float('inf')}
    max_patience = patience
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}] || Learning Rate: {lr_scheduler.get_lr()} || Patience: {patience}")
        
        # TRAINING -------------------------------------------------------------------------------
        model.train()
        
        train_sarcasm_loss, train_sentiment_loss, train_stance_loss = 0.0, 0.0, 0.0    
        correct_sarcasm, correct_sentiment, correct_stance, total_samples = 0,0,0,0

        for batch in tqdm(train_loader, desc="Training"):
            *tweet_data, sent_conf, sarc_conf, stance_conf, sarc_y, sent_y, stance_y = batch
            
            input_ids, attention_mask = [], []
            for i in range(len(tweet_data)//2):
                input_ids.append(tweet_data[i])
            for i in range(len(tweet_data)//2,len(tweet_data)):
                attention_mask.append(tweet_data[i])
            
            for i in range(len(input_ids)):
                input_ids[i] = input_ids[i].to(device)
                attention_mask[i] = attention_mask[i].to(device)
            sarc_y = sarc_y.to(device)
            sent_y = sent_y.to(device)
            stance_y = stance_y.to(device)
            
            optimizer.zero_grad()

            sarcasm_logits, sentiment_logits, stance_logits = None, None, None
            if first_task == "sarcasm":
                sarcasm_logits, sentiment_logits, stance_logits = model(input_ids, attention_mask)
            else:
                sentiment_logits, sarcasm_logits, stance_logits = model(input_ids, attention_mask)

            if WEIGHTING_METHOD == [weighting_settings["PRIORITIZE_HIGH_CONFIDENCE_WEIGHTING"], weighting_settings["PRIORITIZE_LOW_CONFIDENCE_WEIGHTING"]]: 
                model.update_conf_weights(sent_conf, sarc_conf, stance_conf, epoch+1)

            sarcasm_loss = ce_loss(sarcasm_logits, sarc_y) * model.loss_weight[0]
            sentiment_loss = ce_loss(sentiment_logits, sent_y) * model.loss_weight[1]
            stance_loss = ce_loss(stance_logits, stance_y) * model.loss_weight[2]
            
            total_loss = sarcasm_loss + sentiment_loss + stance_loss
            total_loss.backward()
            optimizer.step()
            
            if WEIGHTING_METHOD ==weighting_settings["HEIRARCHICAL_WEIGHTING"]: 
                model.update_hw_weights(sarcasm_loss, sentiment_loss)

            correct_sarcasm += (sarcasm_logits.argmax(dim=1) == sarc_y).sum().item()
            correct_sentiment += (sentiment_logits.argmax(dim=1) == sent_y).sum().item()
            correct_stance += (stance_logits.argmax(dim=1) == stance_y).sum().item()
            total_samples += input_ids[0].size(0)
            
            train_sarcasm_loss += sarcasm_loss.item()
            train_sentiment_loss += sentiment_loss.item()
            train_stance_loss += stance_loss.item()
            
        avg_sarcasm_loss = train_sarcasm_loss / total_samples
        avg_sentiment_loss = train_sentiment_loss / total_samples
        avg_stance_loss = train_stance_loss / total_samples

        sarcasm_acc = correct_sarcasm / total_samples
        sentiment_acc = correct_sentiment / total_samples
        stance_acc = correct_stance / total_samples

        print(f"Sarcasm -> Loss: {avg_sarcasm_loss:.4f}, Acc: {sarcasm_acc:.4f}")
        print(f"Sentiment -> Loss: {avg_sentiment_loss:.4f}, Acc: {sentiment_acc:.4f}")
        print(f"Stance -> Loss: {avg_stance_loss:.4f}, Acc: {stance_acc:.4f}\n")
        
        
        # VALIDATION -------------------------------------------------------------------------------
        model.eval()
        valid_sarcasm_loss, valid_sentiment_loss, valid_stance_loss = 0.0, 0.0, 0.0
        valid_correct_sarcasm, valid_correct_sentiment, valid_correct_stance, valid_total_samples = 0,0,0,0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                *tweet_data, sent_conf, sarc_conf, stance_conf, sarc_y, sent_y, stance_y = batch

                input_ids, attention_mask = [], []
                for i in range(len(tweet_data)//2):
                    input_ids.append(tweet_data[i])
                for i in range(len(tweet_data)//2,len(tweet_data)):
                    attention_mask.append(tweet_data[i])

                for i in range(len(input_ids)):
                    input_ids[i] = input_ids[i].to(device)
                    attention_mask[i] = attention_mask[i].to(device)
                sarc_y = sarc_y.to(device)
                sent_y = sent_y.to(device)
                stance_y = stance_y.to(device)

                sarcasm_logits, sentiment_logits, stance_logits = model(input_ids, attention_mask)

                sarcasm_loss = ce_loss(sarcasm_logits, sarc_y)
                sentiment_loss = ce_loss(sentiment_logits, sent_y)
                stance_loss = ce_loss(stance_logits, stance_y)

                valid_sarcasm_loss += sarcasm_loss.item()
                valid_sentiment_loss += sentiment_loss.item()
                valid_stance_loss += stance_loss.item()

                valid_correct_sarcasm += (sarcasm_logits.argmax(dim=1) == sarc_y).sum().item()
                valid_correct_sentiment += (sentiment_logits.argmax(dim=1) == sent_y).sum().item()
                valid_correct_stance += (stance_logits.argmax(dim=1) == stance_y).sum().item()
                valid_total_samples += input_ids[0].size(0)
        
        avg_valid_sarcasm_loss = valid_sarcasm_loss / valid_total_samples
        avg_valid_sentiment_loss = valid_sentiment_loss / valid_total_samples
        avg_valid_stance_loss = valid_stance_loss / valid_total_samples
        
        valid_sarcasm_acc = valid_correct_sarcasm / valid_total_samples
        valid_sentiment_acc = valid_correct_sentiment / valid_total_samples
        valid_stance_acc = valid_correct_stance / valid_total_samples

        print(f"Sarcasm -> Loss: {avg_valid_sarcasm_loss:.4f}, Acc: {valid_sarcasm_acc:.4f}")
        print(f"Sentiment -> Loss: {avg_valid_sentiment_loss:.4f}, Acc: {valid_sentiment_acc:.4f}")
        print(f"Stance -> Loss: {avg_valid_stance_loss:.4f}, Acc: {valid_stance_acc:.4f}\n\n")

        if valid_stance_acc < best_valid_accs["stance"]:
            if valid_sarcasm_acc < best_valid_accs["sarcasm"]:
                if valid_sentiment_acc < best_valid_accs["sentiment"]:
                    patience -= 1
                else:
                    best_valid_accs["sentiment"] = valid_sentiment_acc
                    patience = max_patience
            else:
                best_valid_accs["sarcasm"] = valid_sarcasm_acc
                patience = max_patience
        else:
            best_valid_accs["stance"] = valid_stance_acc
            patience = max_patience
        
        if patience == 0:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.\n")
            break
        
        lr_scheduler.step()

    return model, best_valid_accs["stance"]

def parallel_trainer(
        train_loader, 
        val_loader, 
        model,
        optimizer, 
        lr_scheduler, 
        ce_loss, 
        num_epochs=20, 
        WEIGHTING_METHOD = weighting_settings["EQUAL_WEIGHTING"],
        patience=5,
        device="cuda",
    ):

    max_patience = patience
    best_valid_accs = {"stance": -float('inf'), "sarcasm": -float('inf'), "sentiment": -float('inf')}
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}] || Learning Rate: {lr_scheduler.get_lr()} || Patience: {patience}")
        
        # TRAINING -------------------------------------------------------------------------------
        model.train()
        
        train_sarcasm_loss, train_sentiment_loss, train_stance_loss = 0.0, 0.0, 0.0    
        correct_sarcasm, correct_sentiment, correct_stance, total_samples = 0,0,0,0

        for batch in tqdm(train_loader, desc="Training"):
            *tweet_data, sent_conf, sarc_conf, stance_conf, sarc_y, sent_y, stance_y = batch
            
            input_ids, attention_mask = [], []
            for i in range(len(tweet_data)//2):
                input_ids.append(tweet_data[i])
            for i in range(len(tweet_data)//2,len(tweet_data)):
                attention_mask.append(tweet_data[i])
            
            for i in range(len(input_ids)):
                input_ids[i] = input_ids[i].to(device)
                attention_mask[i] = attention_mask[i].to(device)
            sarc_y = sarc_y.to(device)
            sent_y = sent_y.to(device)
            stance_y = stance_y.to(device)
            
            optimizer.zero_grad()
            sarcasm_logits, sentiment_logits, stance_logits = model(input_ids, attention_mask)

            if WEIGHTING_METHOD == [weighting_settings["PRIORITIZE_HIGH_CONFIDENCE_WEIGHTING"], weighting_settings["PRIORITIZE_LOW_CONFIDENCE_WEIGHTING"]]: 
                model.update_conf_weights(sent_conf, sarc_conf, stance_conf, epoch+1)

            sarcasm_loss = ce_loss(sarcasm_logits, sarc_y) * model.loss_weight[0]
            sentiment_loss = ce_loss(sentiment_logits, sent_y) * model.loss_weight[1]
            stance_loss = ce_loss(stance_logits, stance_y) * model.loss_weight[2]
            
            total_loss = sarcasm_loss + sentiment_loss + stance_loss
            total_loss.backward()
            optimizer.step()
            
            if WEIGHTING_METHOD ==weighting_settings["HEIRARCHICAL_WEIGHTING"]: 
                model.update_hw_weights(sarcasm_loss, sentiment_loss)

            correct_sarcasm += (sarcasm_logits.argmax(dim=1) == sarc_y).sum().item()
            correct_sentiment += (sentiment_logits.argmax(dim=1) == sent_y).sum().item()
            correct_stance += (stance_logits.argmax(dim=1) == stance_y).sum().item()
            total_samples += input_ids[0].size(0)
            
            train_sarcasm_loss += sarcasm_loss.item()
            train_sentiment_loss += sentiment_loss.item()
            train_stance_loss += stance_loss.item()
            
        avg_sarcasm_loss = train_sarcasm_loss / total_samples
        avg_sentiment_loss = train_sentiment_loss / total_samples
        avg_stance_loss = train_stance_loss / total_samples

        sarcasm_acc = correct_sarcasm / total_samples
        sentiment_acc = correct_sentiment / total_samples
        stance_acc = correct_stance / total_samples

        print(f"Sarcasm -> Loss: {avg_sarcasm_loss:.4f}, Acc: {sarcasm_acc:.4f}")
        print(f"Sentiment -> Loss: {avg_sentiment_loss:.4f}, Acc: {sentiment_acc:.4f}")
        print(f"Stance -> Loss: {avg_stance_loss:.4f}, Acc: {stance_acc:.4f}\n")
        
        
        # VALIDATION -------------------------------------------------------------------------------
        model.eval()
        valid_sarcasm_loss, valid_sentiment_loss, valid_stance_loss = 0.0, 0.0, 0.0
        valid_correct_sarcasm, valid_correct_sentiment, valid_correct_stance, valid_total_samples = 0,0,0,0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                *tweet_data, sent_conf, sarc_conf, stance_conf, sarc_y, sent_y, stance_y = batch

                input_ids, attention_mask = [], []
                for i in range(len(tweet_data)//2):
                    input_ids.append(tweet_data[i])
                for i in range(len(tweet_data)//2,len(tweet_data)):
                    attention_mask.append(tweet_data[i])

                for i in range(len(input_ids)):
                    input_ids[i] = input_ids[i].to(device)
                    attention_mask[i] = attention_mask[i].to(device)
                sarc_y = sarc_y.to(device)
                sent_y = sent_y.to(device)
                stance_y = stance_y.to(device)

                sarcasm_logits, sentiment_logits, stance_logits = model(input_ids, attention_mask)

                sarcasm_loss = ce_loss(sarcasm_logits, sarc_y)
                sentiment_loss = ce_loss(sentiment_logits, sent_y)
                stance_loss = ce_loss(stance_logits, stance_y)

                valid_sarcasm_loss += sarcasm_loss.item()
                valid_sentiment_loss += sentiment_loss.item()
                valid_stance_loss += stance_loss.item()

                valid_correct_sarcasm += (sarcasm_logits.argmax(dim=1) == sarc_y).sum().item()
                valid_correct_sentiment += (sentiment_logits.argmax(dim=1) == sent_y).sum().item()
                valid_correct_stance += (stance_logits.argmax(dim=1) == stance_y).sum().item()
                valid_total_samples += input_ids[0].size(0)
        
        avg_valid_sarcasm_loss = valid_sarcasm_loss / valid_total_samples
        avg_valid_sentiment_loss = valid_sentiment_loss / valid_total_samples
        avg_valid_stance_loss = valid_stance_loss / valid_total_samples
        
        valid_sarcasm_acc = valid_correct_sarcasm / valid_total_samples
        valid_sentiment_acc = valid_correct_sentiment / valid_total_samples
        valid_stance_acc = valid_correct_stance / valid_total_samples

        print(f"Sarcasm -> Loss: {avg_valid_sarcasm_loss:.4f}, Acc: {valid_sarcasm_acc:.4f}")
        print(f"Sentiment -> Loss: {avg_valid_sentiment_loss:.4f}, Acc: {valid_sentiment_acc:.4f}")
        print(f"Stance -> Loss: {avg_valid_stance_loss:.4f}, Acc: {valid_stance_acc:.4f}\n\n")

        if valid_stance_acc < best_valid_accs["stance"]:
            if valid_sarcasm_acc < best_valid_accs["sarcasm"]:
                if valid_sentiment_acc < best_valid_accs["sentiment"]:
                    patience -= 1
                else:
                    best_valid_accs["sentiment"] = valid_sentiment_acc
                    patience = max_patience
            else:
                best_valid_accs["sarcasm"] = valid_sarcasm_acc
                patience = max_patience
        else:
            best_valid_accs["stance"] = valid_stance_acc
            patience = max_patience
        
        if patience == 0:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.\n")
            break
        
        lr_scheduler.step()
    
    return model, best_valid_accs["stance"]