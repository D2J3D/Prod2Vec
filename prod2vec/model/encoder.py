import torch
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F
from transformers import AutoTokenizer, AutoModel


class ProdFeatureEncoder(nn.Module):
    def __init__(self, config):
        '''
                Model for creating embeddings of attributes
        '''
        super(ProdFeatureEncoder, self).__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
        self.fc = nn.Linear(self.config.bert_output_size, self.config.embedding_size)
        init.xavier_uniform_(self.fc.weight)

    def forward(self, text: str):
        t = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        model_output = self.model(**{k: v.to(self.model.device) for k, v in t.items()})
        embedding = model_output.last_hidden_state[:, 0, :]
        embedding = self.fc(embedding)
        return embedding[0]
