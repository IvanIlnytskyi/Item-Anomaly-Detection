import spacy
import os
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, Normalizer, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
import torch
import pandas as pd
from PIL import Image

class DescriptionDataset(Dataset):
    """Dateset for item type classification using item description"""
    def __init__(self, csv_file='../data/products_cleaned.csv'):
        """
        Args:
            csv_file (string): Path to the csv file with items dataframe
        """
        self.csv_file = csv_file
        self.X = []
        self.Y = []
        self.y_codec = LabelEncoder()
        self._init_dataset()
        self.x_len = len(self.X[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), self.Y[idx]

    def _init_dataset(self):
        nlp = spacy.load('en_core_web_md')
        df = pd.read_csv(self.csv_file)[['description', 'category']]
        self.X = [nlp(sentence).vector for sentence in df['description'].to_numpy()]
        self.Y = self.y_codec.fit_transform(df['category'].to_numpy())

# class DescriptionDataset(Dataset):
#     """Dateset for item type classification using item title"""
#     def __init__(self, csv_file='../data/products_cleaned.csv'):
#         """
#         Args:
#             csv_file (string): Path to the csv file with items dataframe
#         """
#         self.csv_file = csv_file
#         self.X = []
#         self.Y = []
#         self.x_codec = Pipeline([('cv', CountVectorizer()), ('tfidf', TfidfTransformer()), ('norm', Normalizer())])
#         self.y_codec = LabelEncoder()
#         self._init_dataset()
#         self.x_len = len(self.X[0])
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         return torch.from_numpy(self.X[idx]).float(), self.Y[idx]
#
#     def _init_dataset(self):
#         df = pd.read_csv(self.csv_file)[['description', 'category']]
#         self.X = self.x_codec.fit_transform(df['description'].to_numpy()).toarray()
#         self.Y = self.y_codec.fit_transform(df['category'].to_numpy())


class TitleDataset(Dataset):
    """Dateset for item type classification using item title"""
    def __init__(self, csv_file='../data/products_cleaned.csv'):
        """
        Args:
            csv_file (string): Path to the csv file with items dataframe
        """
        self.csv_file = csv_file
        self.X = []
        self.Y = []
        self.x_codec = Pipeline([('cv', CountVectorizer()), ('tfidf', TfidfTransformer()), ('norm', Normalizer())])
        self.y_codec = LabelEncoder()
        self._init_dataset()
        self.x_len = len(self.X[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), self.Y[idx]

    def _init_dataset(self):
        df = pd.read_csv(self.csv_file)[['name', 'category']]
        self.X = self.x_codec.fit_transform(df['name'].to_numpy()).toarray()
        self.Y = self.y_codec.fit_transform(df['category'].to_numpy())


class ImageDataset(Dataset):
    """Image dateset for item type classification"""

    def __init__(self, csv_file='../data/products_cleaned.csv', root_dir='../data/img_n/', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with items dataframe
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.Y = []
        self.y_codec = LabelEncoder()
        self.Y = self.y_codec.fit_transform(self.df['category'].to_numpy())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df['id'][idx]+'.jpg')
        image = Image.open(img_name)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, self.Y[idx]


class ExploratoryDataset(Dataset):
    def __init__(self, csv_file='../data/products_cleaned.csv', root_dir='../data/img_n/', transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.Description = []
        self.Title = []
        self.Y = []
        self.title_codec = Pipeline([('cv', CountVectorizer()), ('tfidf', TfidfTransformer()), ('norm', Normalizer())])
        self.y_codec = LabelEncoder()
        self._init_dataset()
        self.title_len = len(self.Title[0])
        self.desc_len = len(self.Description[0])
        self.y_len = len(self.y_codec.classes_)
        self.id = self.df['id'].to_numpy()

    def _init_dataset(self):
        nlp = spacy.load('en_core_web_md')
        self.Description = [nlp(sentence).vector for sentence in self.df['description'].to_numpy()]
        self.Title = self.title_codec.fit_transform(self.df['name'].to_numpy()).toarray()
        self.Y = self.y_codec.fit_transform(self.df['category'].to_numpy())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df['id'][idx]+'.jpg')
        image = Image.open(img_name)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, self.Description[idx], torch.from_numpy(self.Title[idx]).float(), self.Y[idx], self.id[idx]