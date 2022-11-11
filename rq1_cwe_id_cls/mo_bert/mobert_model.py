import torch
import torch.nn as nn


class RobertaClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, feature_token_loc, args=None):
        if isinstance(feature_token_loc, int):
            x = features[:, feature_token_loc, :]
        else:
            x = []
            for i in range(len(feature_token_loc)):
                x.append(features[i, feature_token_loc[i].item(), :].tolist())
            x = torch.tensor(x).to(args.device)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MOBERT(nn.Module):
    def __init__(self, roberta, tokenizer, num_cwe_id, num_cwe_types, args):
        super(MOBERT, self).__init__()
        self.roberta = roberta
        self.tokenizer = tokenizer
        self.args = args
        # FC layer for each group
        self.cls_head_id = RobertaClassificationHead(self.roberta.config, num_cwe_id)
        self.cls_head_type = RobertaClassificationHead(self.roberta.config, num_cwe_types)

    def forward(self, 
               input_ids, 
               cwe_id_labels=None,
               cwe_type_labels=None,
               cwe_label_map=None,
               encoder_last_hidden_state=None,
               return_encoder_last_hidden_state=False):   
        # ... <cls_type>, </s>, <pad>, ..., <pad>
        cls_type_token_loc = torch.ne(input_ids, self.tokenizer.pad_token_id).sum(-1) - 2
        if encoder_last_hidden_state is None:
            encoder_last_hidden_state = self.roberta(input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state
        if return_encoder_last_hidden_state:
            return encoder_last_hidden_state
        cwe_id_logit = self.cls_head_id(encoder_last_hidden_state, feature_token_loc=0)
        cwe_type_logit = self.cls_head_type(encoder_last_hidden_state, feature_token_loc=cls_type_token_loc, args=self.args)
        if cwe_id_labels is not None and cwe_type_labels is not None:
            cls_loss_fct = nn.CrossEntropyLoss()
            cwe_id_loss = cls_loss_fct(cwe_id_logit, cwe_id_labels)
            cwe_type_loss = cls_loss_fct(cwe_type_logit, cwe_type_labels)
            return cwe_id_loss, cwe_type_loss
        else:
            # act the prob for classification tasks
            cwe_id_prob = torch.softmax(cwe_id_logit, dim=-1)
            cwe_type_prob = torch.softmax(cwe_type_logit, dim=-1)
            
            """
            # adjust cwe_id_prob based on cwe_type_prob
            if cwe_label_map is not None:
                weights = []
                for n in range(len(cwe_type_prob)):
                    weight = [0.0 for _ in range(len(cwe_label_map))]
                    i = 0
                    for k, v in cwe_label_map.items():
                        if v[0] == i:
                            weight[i] = cwe_type_prob[n][v[3]]
                        i += 1
                    weights.append(weight)
                weights = torch.tensor(weights).to(self.args.device)
                cwe_id_prob = cwe_id_prob * weights
            """
            return cwe_id_prob, cwe_type_prob