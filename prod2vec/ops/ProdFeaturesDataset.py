import pandas as pd
from torch.utils.data import Dataset


class ProdFeaturesDataset(Dataset):
    def __init__(self, img_filepath, text_filepath, attr_filepath):
        self.img_df = pd.read_parquet(img_filepath)
        self.name_descr_df = pd.read_parquet(text_filepath)
        self.attr_df = pd.read_parquet(attr_filepath)

    def __len__(self):
        return len(self.name_descr_df)

    def __getitem__(self, variant_id):
        main_img_emb = self.img_df.loc[self.img_df['variantid']
                                       == variant_id, 'main_pic_embeddings_resnet_v1']
        name, description = self.name_descr_df.loc[self.name_descr_df['variantid'] == variant_id, [
            'name', 'description']]
        attributes = self.attr_df.loc[self.attr_df['variantid']
                                      == 'variantid', 'characteristic_attributes_mapping']
        return main_img_emb, name, description, attributes
