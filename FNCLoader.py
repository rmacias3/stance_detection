import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FNCDataset(Dataset):
    """Stance Detection dataset."""

    def __init__(self, articles_csv_file, hl_and_stances_csv_file, model):
        """
        Args:
            articles_csv_file (string): Path to the csv file with the article bodies.
            stances_csv_file (string): Path to the csv file with the article headlines and stances.
        """
        self.articles = pd.read_csv(articles_csv_file)
        self.hl_and_stances = pd.read_csv(hl_and_stances_csv_file)
        self.stance_to_idx = {'unrelated': 0, 'agree': 1, 'discuss': 2, 'disagree': 3}
        self.sentence_embedder = model.cuda()
        
    def __len__(self):
        return len(self.articles)
    
    def encode_and_resize(self, sentences):
        #sentence length and paragraph length cut offs chosen based on gpu memory
        sentence_length_cut_off, paragraph_length_cutoff = 3000, 47
        #length chosen to filter out noisy text
        min_sentence_length = 46
        new_sentences = [k for k in sentences if min_sentence_length < len(k) < sentence_length_cut_off]
        #cut_off chosen based on gpu memory limitations
        cut_off = paragraph_length_cutoff if len(new_sentences) > paragraph_length_cutoff else len(new_sentences)
        new_sentences = new_sentences[:cut_off]
        num_sentences = len(new_sentences)
        if not new_sentences:
            new_sentences = ['empty sentence is here!']
            num_sentences = 1
        vecs = self.sentence_embedder.encode(new_sentences, bsize=num_sentences, verbose=False)
        sent_tensors = torch.zeros(num_sentences, 64, 64)
        for i in range(num_sentences):
            sent_tensors[i] = torch.from_numpy(vecs[i].reshape((64, 64)))
        return sent_tensors.view(num_sentences, 1, 64, 64)
    
    def __getitem__(self, idx):
        embedded_article_sentences = self.encode_and_resize(self.articles.iloc[idx]['articleBody'].split('\n'))
        body_id = self.articles.iloc[idx]['Body ID']
        article_stances = self.hl_and_stances.loc[self.hl_and_stances['Body ID'] == body_id] 
        sent_embeddings = self.encode_and_resize(article_stances['Headline'].values)
        pairs = zip(sent_embeddings, [torch.Tensor([self.stance_to_idx[k]]) for k in article_stances['Stance'].values])
        return {'body_embeds': embedded_article_sentences, 'hl_embeds_and_stance' : pairs}
