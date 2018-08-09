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
    def __init__(self, path_folder, transform=None, sequence_length=10):
        self.origin_corpus = Texts(path_folder)
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.origin_corpus.valid) - self.sequence_length

    def __getitem__(self, idx):

        end = idx + self.sequence_length

        id_symbols_x = self.origin_corpus.valid[idx:end]
        id_symbols_y = self.origin_corpus.valid[end + 1]

        if self.transform:
            # Буквы перводить в id или в one-hot вектора
            id_symbols_x = self.transform(id_symbols_x, self.origin_corpus.dictionary)
            id_symbols_y = self.transform(id_symbols_y, self.origin_corpus.dictionary)[0]
        sample = {'serial': id_symbols_x, 'predict_letter': id_symbols_y}
        return sample


class ToSymbols(object):
    """
    id -> letter
    """
    def __call__(self, x, dic):
        output = []
        if len(x.size()) != 0:
            for item in x:
                output.append(dic.idx2symbol[int(item)])
        else:
            output.append(dic.idx2symbol[int(x)])
        return output


class ToOneHot(object):
    """
    id -> one hot vector
    """
    def __call__(self, x, dic):
        output = []
        if len(x.size()) != 0:
            for item in x:
                output.append([1 if i == item else 0 for i in range(len(dic))])
        else:
            output.append([1 if i == int(x) else 0 for i in range(len(dic))])
        return output


class Nothing(object):
    def __call__(self, x):
        return x

if __name__ == "__main__":

    #wikitextDataset = WikitextDataset('wikitext', transform=ToSymbols())  # сэмлы из символов

    wikitextDataset = WikitextDataset('wikitext', transform=ToOneHot())  # сэмлы из one_hot векторов

    dataloader = DataLoader(wikitextDataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=Nothing())

    # Покажет первые 10 бачей
    for i_batch, sample_batched in enumerate(dataloader):
        for one_batch in sample_batched:
            print(one_batch)
        print("-"*10)
        if (i_batch + 1) % 10 == 0:
            break

    print("Done!")

