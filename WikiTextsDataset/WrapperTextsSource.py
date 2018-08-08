import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.NN.WikiTextsDataset.wiki_utils import Texts

wikitext_folder = os.path.abspath(os.path.join(os.path.curdir, 'wikitext'))


class WikitextDataset(Dataset):
    """
    Dataset of Wikitext
    """
    def __init__(self, path_folder, transform=None, sequence_length=30):
        self.origin_corpus = Texts(path_folder)
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.origin_corpus.valid)

    def __getitem__(self, idx):

        end = idx + self.sequence_length

        if end + 1 > len(self):
            raise IndexError()

        id_symbols_x = self.origin_corpus.valid[idx:end]
        id_symbols_y = self.origin_corpus.valid[end + 1]

        if self.transform:
            # Буквы перводить в id или в one-hot вектора
            id_symbols_x = self.transform(id_symbols_x)
            id_symbols_y = self.transform(id_symbols_y)

        return id_symbols_x, id_symbols_y



if __name__ == "__main__":

    wikitextDataset = WikitextDataset('wikitext')

    dataloader = DataLoader(wikitextDataset, batch_size=4, shuffle=True, num_workers=1)

    for i_batch, sample_batched in enumerate(dataloader):
        print(sample_batched)

