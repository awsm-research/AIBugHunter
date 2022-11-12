import torch
import torch.nn as nn


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BERT(nn.Module):
    def __init__(self, roberta, tokenizer, args):
        super(BERT, self).__init__()
        self.roberta = roberta
        self.tokenizer = tokenizer
        self.args = args
        # FC layer for each group
        self.cwe_type_cls = RobertaClassificationHead(self.roberta.config)
        
    def forward(self, 
               input_ids, 
               cvss_labels=None):
        last_hidden_state = self.roberta(input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state
        cvss_logit = self.cwe_type_cls(last_hidden_state)
        if cvss_labels is not None:
            reg_loss_fct = nn.MSELoss()
            cvss_loss = reg_loss_fct(cvss_logit, cvss_labels)
            return cvss_loss
        return cvss_logit