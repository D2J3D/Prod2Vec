import torch
from torch import nn

from model.encoder import ProdFeatureEncoder


class Prod2VecModel(nn.Module):
    """
       A PyTorch model that combines image, attribute, name,
       and description embeddings to produce a product embedding.

       Attributes:
           img_encoder (nn.Linear): Projects image embeddings to the embedding size.
           attr_encoder (nn.Sequential): a ProdFeatureEncoder and a linear layer.
           name_encoder (nn.Sequential): a ProdFeatureEncoder and a linear layer.
           descr_encoder (nn.Sequential): a ProdFeatureEncoder and a linear layer.
           bn (nn.BatchNorm1d): A batch normalization layer for the concatenated embeddings.
    """

    def __init__(self, mode, config):
        """
            Args:
            :param mode (str): The mode of the model, one of 'train', 'eval', or 'inference'.
            :param config: A configuration object containing hyperparameters and settings.
        """
        super().__init__()
        self.mode = mode
        assert mode in ['train', 'eval', 'inference']
        self.config = config

        self.img_encoder = nn.Linear(
            self.config.img_embedding_size, self.config.embedding_size)

        self.attr_encoder = nn.Sequential(
            ProdFeatureEncoder(self.config),
            nn.Linear(self.config.bert_output_size,
                      self.config.embedding_size),
        )

        self.name_encoder = nn.Sequential(
            ProdFeatureEncoder(self.config),
            nn.Linear(self.bert_output_size, self.config.embedding_size),
        )

        self.descr_encoder = nn.Sequential(
            ProdFeatureEncoder(self.config),
            nn.Linear(self.bert_output_size, self.config.embedding_size)
        )

        self.bn = nn.BatchNorm1d(self.embedding_size*4)

    def forward(self, img_embedding, name, descr, attr):
        """
            Forward pass of the model.

            Args:
                img_embedding (Tensor): The image embedding.
                name (Tensor): The name feature.
                descr (Tensor): The description feature.
                attr (Tensor): The attribute feature.

            Returns:
                A tuple of five tensors:
                    - img_embedding (Tensor): The projected image embedding.
                    - attr_embedding (Tensor): The encoded attribute embedding.
                    - name_embedding (Tensor): The encoded name embedding.
                    - descr_embedding (Tensor): The encoded description embedding.
                    - product_embedding (Tensor): The concatenated and normalized product embedding.
        """
        img_embedding = self.img_encoder(img_embedding)
        attr_embedding = self.attr_encoder(attr)
        name_embedding = self.name_encoder(name)
        descr_embedding = self.descr_encoder(descr)

        product_embedding = torch.cat(
            (img_embedding, attr_embedding, name_embedding, descr_embedding), 1)
        product_embedding = self.bn(product_embedding)
        return {
            'img': img_embedding,
            'attr': attr_embedding,
            'name': name_embedding,
            'descr': descr_embedding,
            'product': product_embedding
        }

    def get_loss(self, model_output, ground_truth):
        return
