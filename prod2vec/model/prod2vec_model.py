import torch.nn as nn

from encoder import ProdFeatureEncoder


class Prod2VecModel(nn.Module):
    def __init__(self, mode, config):
        super(Prod2VecModel, self).__init__()
        self.mode = mode
        assert mode in ['train', 'eval', 'inference']
        self.config = config

        self.img_encoder = nn.Linear(self.config.img_embedding_size, self.config.embedding_size)

        self.attr_encoder = nn.Sequential(
            ProdFeatureEncoder(self.config),
            nn.Linear(self.config.bert_output_size, self.config.embedding_size),
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
        img_embedding = self.img_encoder(img_embedding)
        attr_embedding = self.attr_encoder(attr)
        name_embedding = self.name_encoder(name)
        descr_embedding = self.descr_encoder(descr)

        product_embedding = torch.cat((img_embedding, attr_embedding, name_embedding, descr_embedding), axis=1)
        product_embedding = self.bn(product_embedding)
        return img_embedding, attr_embedding, name_embedding, descr_embedding, product_embedding
