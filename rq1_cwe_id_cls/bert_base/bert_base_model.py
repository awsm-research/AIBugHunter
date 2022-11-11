import torch
import torch.nn as nn


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BERT(nn.Module):
    def __init__(self, roberta, tokenizer, num_cwe_types, args):
        super(BERT, self).__init__()
        self.roberta = roberta
        self.tokenizer = tokenizer
        self.args = args
        # FC layer for each group
        self.cwe_type_cls = RobertaClassificationHead(self.roberta.config, num_cwe_types)
        
    def forward(self, 
               input_ids, 
               cwe_type_labels=None,
               logit_adjustment=None):
        last_hidden_state = self.roberta(input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state
        cwe_type_logit = self.cwe_type_cls(last_hidden_state)
        if cwe_type_labels is not None:
            cls_loss_fct = nn.CrossEntropyLoss()
            cwe_type_loss = cls_loss_fct(cwe_type_logit, cwe_type_labels)
            return cwe_type_loss
        else:
            # act the prob for classification tasks
            cwe_type_prob = torch.softmax(cwe_type_logit, dim=-1)
        return cwe_type_prob