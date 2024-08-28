import torch
import torch.nn as nn

from config.config import ModelConfig
from model.encoder import ProdFeatureEncoder


if __name__ == "__main__":
    model_config = ModelConfig()
    encoder = ProdFeatureEncoder(model_config)
    attributes_input = """
    {'Цвет товара': ['серый'],
     'Ширина, см': ['0.8'],
     'Бренд': ['Prym'],
     'Тип': ['Тесьма'],
     'Состав ниток': ['Ткань'],
     'Страна-изготовитель': ['Германия'],
     'Длина, м': ['2']}
    """
    encoded_input=encoder(attributes_input)
    print(encoded_input)
