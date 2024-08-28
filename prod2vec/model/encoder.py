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
        self.fc = nn.Linear(self.config.bert_output_size, self.config.attr_embedding_size)
        init.xavier_uniform_(self.fc.weight)

    def forward(self, attribute_string: str):
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
            embeddings = model_output.last_hidden_state[:, 0, :]
            embeddings = self.fc(embeddings)

        embeddings = F.normalize(embeddings)
        return embeddings[0].cpu().numpy()
