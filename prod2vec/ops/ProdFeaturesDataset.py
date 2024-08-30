import pandas as pd
import torch
from torch.utils.data import Dataset


class ProdFeaturesDataset(Dataset):
    """
    A PyTorch Dataset class for loading and transforming product features.
    """

    def __init__(self, img_filepath, text_filepath, attr_filepath, text_transform):
        """
        Parameters
        ---------
        :param img_filepath : str
                Filepath to the parquet file containing image embeddings.
        :param text_filepath : str
                Filepath to the parquet file containing name and description data.
                **This DataFrame must not contain any NaN values** and its index should be in ascending order.
        :param attr_filepath : str
                Filepath to the parquet file containing attribute data.
                **This DataFrame must not contain any NaN values** and its index should be in ascending order.
        :param text_transform : callable
                A function that takes in a string and returns a transformed string.
        """
        self.img_df = pd.read_parquet(img_filepath)
        self.name_descr_df = pd.read_parquet(text_filepath)
        self.attr_df = pd.read_parquet(attr_filepath)
        self.text_transform = text_transform

        # ensure everything is ok with passed parquet data
        self.img_df = self.img_df.reset_index(drop=True)
        self.name_descr_df = self.name_descr_df.reset_index(drop=True)
        self.attr_df = self.attr_df.reset_index(drop=True)

        self.name_descr_df = self.name_descr_df.fillna("")
        self.attr_df = self.attr_df.fillna("")

    def __len__(self):
        return len(self.name_descr_df)

    def __getitem__(self, idx):
        main_img_emb = self.img_df.loc[idx, 'main_pic_embeddings_resnet_v1']
        main_img_emb = torch.tensor(main_img_emb[0])
        name, description = self.name_descr_df.loc[idx, [
            'name', 'description']]
        name, description = self.text_transform(
            name), self.text_transform(description)
        attributes = self.text_transform(
            self.attr_df.loc[idx, 'characteristic_attributes_mapping'])
        return main_img_emb, name, description, attributes
