import torch
from torch import nn
from torch.nn import init
from transformers import AutoTokenizer, AutoModel


class ProdFeatureEncoder(nn.Module):
    """
    Model for creating embeddings with pre-trained ruBERT-tiny BERT.

    Attributes:
        config (object): Configuration object containing model hyperparameters.
        tokenizer (AutoTokenizer): Tokenizer instance for ruBERT-tiny.
        model (AutoModel): Pre-trained ruBERT-tiny model instance.
        fc (nn.Linear): Linear layer for dimensionality reduction.
    """
    def __init__(self, config):
        """
        Initializes the ProdFeatureEncoder model.

        Args:
            config (object): Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
        self.fc = nn.Linear(self.config.bert_output_size, self.config.embedding_size)
        init.xavier_uniform_(self.fc.weight)
        self.norm = nn.LayerNorm(self.config.embedding_size)

    def forward(self, text: str):
        """
        Creates an embedding for the input text.
        Args:
            text (str): Input text to create an embedding for.
        Returns:
            torch.Tensor: Embedding vector for the input text.
        """
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        model_output = self.model(**{k: v.to(self.model.device) for k, v in tokens.items()})
        embedding = model_output.last_hidden_state[:, 0, :]
        embedding = self.fc(embedding)
        return embedding[0]
