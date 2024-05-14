import os
import os
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap
import torch.nn.functional as F
from transformers import AutoModel
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Some weights of*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ensemble_settings = {
    "ENSEMBLE_AVERAGE" : 0,
    "ENSEMBLE_ATTENTION" : 1,
    "ENSEMBLE_GRAPH" : 2
}

weighting_settings = {
    "EQUAL_WEIGHTING" : 0,
    "STATIC_WEIGHTING" : 1,
    "RELATIVE_WEIGHTING" : 2,
    "HEIRARCHICAL_WEIGHTING" : 3,
    "PRIORITIZE_HIGH_CONFIDENCE_WEIGHTING" : 4,
    "PRIORITIZE_LOW_CONFIDENCE_WEIGHTING" : 5,
    "META_WEIGHTING": 6
}

class SequentialTaskHead(nn.Module):
    def __init__(self, num_labels, hidden_size, task_name, is_first_head=False, pool_bert_output=False, use_bi=False, use_gru=False):
        super(SequentialTaskHead, self).__init__()

        self.num_labels = num_labels
        self.pool_bert_output = pool_bert_output
        
        if not pool_bert_output:
            multiplier_for_sequential = 1
            if not is_first_head: multiplier_for_sequential += 1
            if not pool_bert_output and use_bi and not is_first_head: multiplier_for_sequential += 1

            if use_gru:
                print(f"Initializing GRU for {task_name} head with bi={use_bi}")
                self.sequential_model = nn.GRU(hidden_size * multiplier_for_sequential, hidden_size, batch_first=True, bidirectional=use_bi)
            else:
                print(f"Initializing LSTM for {task_name} head with bi={use_bi}")
                self.sequential_model = nn.LSTM(hidden_size * multiplier_for_sequential, hidden_size, batch_first=True, bidirectional=use_bi)

        multiplier_for_classifier = 1
        if not pool_bert_output and use_bi: multiplier_for_classifier += 1
        if pool_bert_output and not is_first_head: multiplier_for_classifier += 1

        print(f"Initializing classifier for {task_name} head, is_first_head = {is_first_head}")
        self.hidden = nn.Linear(hidden_size * multiplier_for_classifier, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

        print()

    def forward(self, inputs):
        if not self.pool_bert_output:
            inputs, _ = self.sequential_model(inputs)
        
        logits = self.hidden(F.dropout(inputs if self.pool_bert_output else inputs[:, -1, :], p=0.1, training=self.training))
        res = self.classifier(logits)
        
        return logits if self.pool_bert_output else inputs, res
    
    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()


class ParallelTaskHead(nn.Module):
    def __init__(self, num_labels, hidden_size, task_name, pool_bert_output=False, use_bi=False, use_gru=False):
        super(ParallelTaskHead, self).__init__()
        
        self.pool_bert_output = pool_bert_output
        if not pool_bert_output:
            if use_gru:
                print(f"Initializing GRU for {task_name} head with bi={use_bi}")
                self.sequential_model = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=use_bi)
            else:
                print(f"Initializing LSTM for {task_name} head with bi={use_bi}")
                self.sequential_model = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=use_bi)

        self.num_labels = num_labels

        print(f"Initializing classifier for {task_name} head")
        multiplier = 2 if use_bi and not pool_bert_output else 1
        self.classifier = nn.Linear(hidden_size * multiplier, num_labels)
        
    def forward(self, inputs):
        if not self.pool_bert_output:
            inputs, _ = self.sequential_model(inputs)
            inputs = inputs[:, -1, :]

        logits = F.dropout(inputs, p=0.1, training=self.training)
        logits = self.classifier(logits)
        
        return logits
    
    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()


class SequentialMultiTaskModel(nn.Module):
    def __init__(
            self, bert_models, 
            sentiment_head, 
            sarcasm_head, 
            stance_head, 
            subtask_hidden_layer_size, 
            combination_method=None, 
            weighting_method=weighting_settings["EQUAL_WEIGHTING"],
            first_task="sentiment",
            pool_bert_output=False,
            device = "cuda",
        ):

        super(SequentialMultiTaskModel, self).__init__()

        print("Initialized Sequential Multi-Task Model\n")

        self.device = device
        self.pool_bert_output = pool_bert_output
        self.first_task = first_task    

        self.combination_method = combination_method
        self.bert_models = nn.ModuleList([AutoModel.from_pretrained(bert).to(device) for bert in bert_models])
                        
        self.sentiment_head = sentiment_head
        self.sarcasm_head = sarcasm_head
        self.stance_head = stance_head

        self.weighting_method = weighting_method
        self.combination_method = combination_method
        self.subtask_hidden_layer_size = subtask_hidden_layer_size

        self.intialize_ensemble_specific_components()
        self.initialize_loss_specific_components()

    def intialize_ensemble_specific_components(self):
        if self.combination_method == ensemble_settings["ENSEMBLE_GRAPH"]:
            graph_hidden_size = self.get_hidden_in()
            self.conv1 = GCNConv(graph_hidden_size, graph_hidden_size)

        if self.combination_method == ensemble_settings["ENSEMBLE_ATTENTION"]:
            self.attention_weights = nn.Linear(self.get_hidden_in(), 1)

    def initialize_loss_specific_components(self):
        if self.weighting_method == weighting_settings["EQUAL_WEIGHTING"]:
            self.loss_weight = [1,1,1]
        if self.weighting_method == weighting_settings["STATIC_WEIGHTING"]:
            self.loss_weight = [0.1, 0.3, 0.6]
        if self.weighting_method == weighting_settings["RELATIVE_WEIGHTING"]:
            self.loss_weight = [1/3, 1/2, 1]
        if self.weighting_method == weighting_settings["HEIRARCHICAL_WEIGHTING"]:
            self.loss_weight = [1,1,1]
        if self.weighting_method in [weighting_settings["PRIORITIZE_HIGH_CONFIDENCE_WEIGHTING"], weighting_settings["PRIORITIZE_LOW_CONFIDENCE_WEIGHTING"]]:
            self.loss_weight = [1,1,1]
        if self.weighting_method == weighting_settings["META_WEIGHTING"]:
            self.meta_weight_layer = nn.Linear(self.subtask_hidden_layer_size, 3)
            self.loss_weight = [1,1,1]

    
    def get_hidden_in(self):
        if len(self.bert_models) == 1:
            return self.bert_models[0].config.hidden_size
        if self.combination_method == ensemble_settings["ENSEMBLE_AVERAGE"]:
            return self.bert_models[0].config.hidden_size
        if self.combination_method == ensemble_settings["ENSEMBLE_ATTENTION"]:
            return self.bert_models[0].config.hidden_size
        if self.combination_method == ensemble_settings["ENSEMBLE_GRAPH"]:
            return self.bert_models[0].config.hidden_size

    def average(self, bert_outputs):
        return torch.mean(torch.stack(bert_outputs), dim=0)

    def attention_aggregate(self, bert_outputs):
        bert_outputs = torch.stack(bert_outputs, dim=1)
        bert_outputs_transposed = bert_outputs.transpose(0, 1)
        attention_weights = F.softmax(self.attention_weights(bert_outputs_transposed), dim=0)
        aggregated_output = torch.mean(bert_outputs_transposed * attention_weights, dim=0)
        
        return aggregated_output

    def graph_aggregate(self, bert_outputs):
        graph_batch = []
        batch_size = bert_outputs[0].shape[0]

        for i in range(batch_size):
            node_features = torch.stack([bert_outputs[j][i] for j in range(len(bert_outputs))])
            edges = []
            for i in range(len(node_features)):
                for j in range(len(node_features)):
                    if i != j:
                        edges.append([i, j])
                        
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            graph = Data(x=node_features, edge_index=edge_index).to(self.device)

            graph_batch.append(graph)

        x = torch.stack([graph.x for graph in graph_batch], dim=0)
        edge_index = torch.cat([graph.edge_index for graph in graph_batch], dim=1)
        graph_batch = Data(x=x, edge_index=edge_index).to(self.device)

        x = F.tanh(self.conv1(graph_batch.x, graph_batch.edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = gap(x, graph_batch.batch)
        
        return x
            
    def update_hw_weights(self, sarcasm_loss, sentiment_loss):
        if self.weighting_method == weighting_settings["HEIRARCHICAL_WEIGHTING"]:
            self.loss_weight[2] = max(min((sarcasm_loss.item() / sentiment_loss.item()) * self.loss_weight[2], 1), 0.01)

    def update_conf_weights(self, sent_conf, sarc_conf, stance_conf, epoch):
        mean_sarc_conf = sum(sarc_conf).item()/len(sarc_conf)
        mean_sent_conf = sum(sent_conf).item()/len(sent_conf)
        mean_stance_conf = sum(stance_conf).item()/len(stance_conf)

        if self.weighting_method == weighting_settings["PRIORITIZE_LOW_CONFIDENCE_WEIGHTING"]:
            self.loss_weight[0] = 1 + ((1 - mean_sarc_conf)/(epoch * 0.5))
            self.loss_weight[1] = 1 + ((1 - mean_sent_conf)/(epoch * 0.5))
            self.loss_weight[2] = 1 + ((1 - mean_stance_conf)/(epoch * 0.5))

        if self.weighting_method == weighting_settings["PRIORITIZE_HIGH_CONFIDENCE_WEIGHTING"]:
            self.loss_weight[0] = 1 + ((mean_sarc_conf)/(epoch * 0.5))
            self.loss_weight[1] = 1 + ((mean_sent_conf)/(epoch * 0.5))
            self.loss_weight[2] = 1 + ((mean_stance_conf)/(epoch * 0.5))

    def forward(self, input_ids, attention_mask):
        outputs = []
        
        for i, bert in enumerate(self.bert_models):
            if self.pool_bert_output:
                outputs.append(bert(input_ids=input_ids[i], attention_mask=attention_mask[i]).last_hidden_state[:, 0, :])
            else:
                outputs.append(bert(input_ids=input_ids[i], attention_mask=attention_mask[i]).last_hidden_state)

        combined_emb = outputs[0]
        if len(self.bert_models) > 1:
            if self.combination_method == ensemble_settings["ENSEMBLE_AVERAGE"]:
                combined_emb = self.average(outputs)
            if self.combination_method == ensemble_settings["ENSEMBLE_ATTENTION"]:
                combined_emb = self.attention_aggregate(outputs)
            if self.combination_method == ensemble_settings["ENSEMBLE_GRAPH"]:
                combined_emb = self.graph_aggregate(outputs)
        
        sarcasm_pred, sentiment_pred, stance_pred = None, None, None
        if self.first_task == "sarcasm":
            sarcasm_logits, sarcasm_pred = self.sarcasm_head(combined_emb)
            input_to_sentiment_head = torch.cat([combined_emb, sarcasm_logits], dim=-1) 

            sentiment_logits, sentiment_pred = self.sentiment_head(input_to_sentiment_head)
            input_to_stance_head = torch.cat([combined_emb, sentiment_logits], dim=-1)

            _, stance_pred = self.stance_head(input_to_stance_head)
        else:
            sentiment_logits, sentiment_pred = self.sentiment_head(combined_emb)
            input_to_sarcasm_head = torch.cat([combined_emb, sentiment_logits], dim=-1)
            sarcasm_logits, sarcasm_pred = self.sarcasm_head(input_to_sarcasm_head)
            input_to_stance_head = torch.cat([combined_emb, sentiment_logits], dim=-1)

            _, stance_pred = self.stance_head(input_to_stance_head)
        
        meta_weights = None
        if self.weighting_method == weighting_settings["META_WEIGHTING"]:
            meta_weights = self.meta_weight_layer(combined_emb)
            meta_weights = F.softmax(meta_weights, dim=-1)
            self.loss_weight = meta_weights.mean(dim=0).tolist()
        
        return sarcasm_pred, sentiment_pred, stance_pred
    

class ParrallelMultiTaskModel(nn.Module):
    def __init__(
            self, 
            bert_models, 
            sentiment_head, 
            sarcasm_head, 
            stance_head, 
            subtask_hidden_layer_size, 
            combination_method=None, 
            weighting_method=weighting_settings["EQUAL_WEIGHTING"], 
            pool_bert_output=False,
            device = "cuda"
        ):

        super(ParrallelMultiTaskModel, self).__init__()

        print("Initialized Parallel Multi-Task Model\n")

        self.device = device
        self.pool_bert_output = pool_bert_output
        
        self.combination_method = combination_method
        self.bert_models = nn.ModuleList([AutoModel.from_pretrained(bert).to(device) for bert in bert_models])
                        
        self.sentiment_head = sentiment_head
        self.sarcasm_head = sarcasm_head
        self.stance_head = stance_head

        self.weighting_method = weighting_method
        self.combination_method = combination_method
        self.subtask_hidden_layer_size = subtask_hidden_layer_size

        self.intialize_ensemble_specific_components()
        self.initialize_loss_specific_components()

    def intialize_ensemble_specific_components(self):
        if self.combination_method == ensemble_settings["ENSEMBLE_GRAPH"]:
            graph_hidden_size = self.get_hidden_in()
            self.conv1 = GCNConv(graph_hidden_size, graph_hidden_size)

        if self.combination_method == ensemble_settings["ENSEMBLE_ATTENTION"]:
            self.attention_weights = nn.Linear(self.get_hidden_in(), 1)

    def initialize_loss_specific_components(self):
        if self.weighting_method == weighting_settings["EQUAL_WEIGHTING"]:
            self.loss_weight = [1,1,1]
        if self.weighting_method == weighting_settings["STATIC_WEIGHTING"]:
            self.loss_weight = [0.1, 0.3, 0.6]
        if self.weighting_method == weighting_settings["RELATIVE_WEIGHTING"]:
            self.loss_weight = [1/3, 1/2, 1]
        if self.weighting_method == weighting_settings["HEIRARCHICAL_WEIGHTING"]:
            self.loss_weight = [1,1,1]
        if self.weighting_method in [weighting_settings["PRIORITIZE_HIGH_CONFIDENCE_WEIGHTING"], weighting_settings["PRIORITIZE_LOW_CONFIDENCE_WEIGHTING"]]:
            self.loss_weight = [1,1,1]
        if self.weighting_method == weighting_settings["META_WEIGHTING"]:
            self.meta_weight_layer = nn.Linear(self.subtask_hidden_layer_size, 3)
            self.loss_weight = [1,1,1]

    
    def get_hidden_in(self):
        if len(self.bert_models) == 1:
            return self.bert_models[0].config.hidden_size
        if self.combination_method == ensemble_settings["ENSEMBLE_AVERAGE"]:
            return self.bert_models[0].config.hidden_size
        if self.combination_method == ensemble_settings["ENSEMBLE_ATTENTION"]:
            return self.bert_models[0].config.hidden_size
        if self.combination_method == ensemble_settings["ENSEMBLE_GRAPH"]:
            return self.bert_models[0].config.hidden_size

    def average(self, bert_outputs):
        return torch.mean(torch.stack(bert_outputs), dim=0)

    def attention_aggregate(self, bert_outputs):
        bert_outputs = torch.stack(bert_outputs, dim=1)
        bert_outputs_transposed = bert_outputs.transpose(0, 1)
        attention_weights = F.softmax(self.attention_weights(bert_outputs_transposed), dim=0)
        aggregated_output = torch.mean(bert_outputs_transposed * attention_weights, dim=0)
        
        return aggregated_output

    def graph_aggregate(self, bert_outputs):
        graph_batch = []
        batch_size = bert_outputs[0].shape[0]

        for i in range(batch_size):
            node_features = torch.stack([bert_outputs[j][i] for j in range(len(bert_outputs))])
            edges = []
            for i in range(len(node_features)):
                for j in range(len(node_features)):
                    if i != j:
                        edges.append([i, j])
                        
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            graph = Data(x=node_features, edge_index=edge_index).to(self.device)

            graph_batch.append(graph)

        x = torch.stack([graph.x for graph in graph_batch], dim=0)
        edge_index = torch.cat([graph.edge_index for graph in graph_batch], dim=1)
        graph_batch = Data(x=x, edge_index=edge_index).to(self.device)

        x = F.tanh(self.conv1(graph_batch.x, graph_batch.edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = gap(x, graph_batch.batch)
        
        return x
            
    def update_hw_weights(self, sarcasm_loss, sentiment_loss):
        if self.weighting_method == weighting_settings["HEIRARCHICAL_WEIGHTING"]:
            self.loss_weight[2] = max(min((sarcasm_loss.item() / sentiment_loss.item()) * self.loss_weight[2], 2), 1)

    def update_conf_weights(self, sent_conf, sarc_conf, stance_conf, epoch):
        mean_sarc_conf = sum(sarc_conf).item()/len(sarc_conf)
        mean_sent_conf = sum(sent_conf).item()/len(sent_conf)
        mean_stance_conf = sum(stance_conf).item()/len(stance_conf)

        if self.weighting_method == weighting_settings["PRIORITIZE_LOW_CONFIDENCE_WEIGHTING"]:
            self.loss_weight[0] = 1 + ((1 - mean_sarc_conf)/(epoch * 0.5))
            self.loss_weight[1] = 1 + ((1 - mean_sent_conf)/(epoch * 0.5))
            self.loss_weight[2] = 1 + ((1 - mean_stance_conf)/(epoch * 0.5))

        if self.weighting_method == weighting_settings["PRIORITIZE_HIGH_CONFIDENCE_WEIGHTING"]:
            self.loss_weight[0] = 1 + ((mean_sarc_conf)/(epoch * 0.5))
            self.loss_weight[1] = 1 + ((mean_sent_conf)/(epoch * 0.5))
            self.loss_weight[2] = 1 + ((mean_stance_conf)/(epoch * 0.5))

    def forward(self, input_ids, attention_mask):
        outputs = []
        
        for i, bert in enumerate(self.bert_models):
            if self.pool_bert_output:
                outputs.append(bert(input_ids=input_ids[i], attention_mask=attention_mask[i]).last_hidden_state[:, 0, :])
            else:
                outputs.append(bert(input_ids=input_ids[i], attention_mask=attention_mask[i]).last_hidden_state)

        combined_emb = outputs[0]
        if len(self.bert_models) > 1:
            if self.combination_method == ensemble_settings["ENSEMBLE_AVERAGE"]:
                combined_emb = self.average(outputs)
            if self.combination_method == ensemble_settings["ENSEMBLE_ATTENTION"]:
                combined_emb = self.attention_aggregate(outputs)
            if self.combination_method == ensemble_settings["ENSEMBLE_GRAPH"]:
                combined_emb = self.graph_aggregate(outputs)
        
        sarcasm_logits = self.sarcasm_head(combined_emb)
        sentiment_logits = self.sentiment_head(combined_emb)
        stance_logits = self.stance_head(combined_emb)
        
        meta_weights = None
        if self.weighting_method == weighting_settings["META_WEIGHTING"]:
            meta_weights = self.meta_weight_layer(combined_emb)
            meta_weights = F.softmax(meta_weights, dim=-1)
            self.loss_weight = meta_weights.mean(dim=0).tolist()
        
        return sarcasm_logits, sentiment_logits, stance_logits