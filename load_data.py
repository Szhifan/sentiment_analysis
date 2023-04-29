import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torchtext
from transformers import AutoTokenizer
import numpy as np
from datasets import load_dataset,Dataset,disable_caching
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import re 

disable_caching()

class BertDataset():
    def __init__(self,dir) -> None:
            
        df = pd.read_csv(dir,encoding='latin1')
        self.tokenizer =  AutoTokenizer.from_pretrained("bert-base-uncased")
        self.dataset = Dataset.from_pandas(df)
        self.id2label = {i:j for i,j in enumerate(set(self.dataset["label"]))}
        self.labels2id = {j:i for i,j in enumerate(set(self.dataset["label"]))}
        self.dataset = self.dataset.map(self._pre_process)
        

    def _pre_process(self,example):
        text = example["text"] 
        encoding = self.tokenizer(text) 
        label = example["label"]
        example["encoding"] = encoding
        example["label_id"] = self.labels2id[label]
        return example

    def get_dataset(self,split=0.2):
        enc_dataset = self.dataset.with_format(type='torch', columns=['label_id',"encoding"])
    
        if split is None:
            return enc_dataset 
        data = enc_dataset.train_test_split(split)
        train_data = data["train"]
        test_data = data["test"]
        return train_data,test_data
    def collate(self,batch):
        ids = [torch.LongTensor(i['encoding']["input_ids"]) for i in batch]
        batch_length = torch.tensor([len(i) for i in ids])
        ids = nn.utils.rnn.pad_sequence(ids, padding_value=self.tokenizer.pad_token_id, batch_first=True)
        label_id = torch.tensor([i['label_id'] for i in batch])
        batch = {'ids': ids,
                'lens': batch_length,
                'label_id': label_id}
        return batch
        
class DefaultDataset():
    def __init__(self,dir) -> None:
        df = pd.read_csv(dir,encoding='latin1')
        self.dataset = Dataset.from_pandas(df)
        self.id2label = {i:j for i,j in enumerate(set(self.dataset["label"]))}
        self.labels2id = {j:i for i,j in enumerate(set(self.dataset["label"]))}
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        self.dataset = self.dataset.map(self._pre_process)
        self.dictionary = self._get_dict()
    def _pre_process(self,sample):
        def match_and_sub(t:str):
            t = re.sub(r"@\S+","",t) #remove @ 
            t = re.sub(r"http:\/\/\S+","",t) #remove links
            # match non-emoji and non alphabet
            t = re.sub(r"\n","",t) #sub newline 
            t = re.sub(r"[^\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FFa-zA-Z\s!\?]","",t)
            t = t.strip()
            return t.lower() 
        text = sample["text"]
        text = match_and_sub(text)
        tokens = self.tokenizer(text)
        label = sample["label"]
        label_id = self.labels2id[label]
        lens = len(tokens)
    
        return {"tokens":tokens,"lens":lens,"label_id":label_id}
    def _get_dict(self):
        special_tokens = ['<unk>', '<pad>']
        dictionary = torchtext.vocab.build_vocab_from_iterator(self.dataset["tokens"],
                                                    min_freq=3,
                                                    specials=special_tokens)
        unk_id = dictionary["<unk>"]
        pad_id = dictionary["<pad>"]
        dictionary.set_default_index(unk_id)
        return dictionary
    def _binarize(self,sample):
        ids = [self.dictionary[token] for token  in sample["tokens"]]
        return {"ids":ids}
    def get_dataset(self,split=0.2):
        dataset = self.dataset.map(self._binarize)
        dataset = dataset.with_format(type='torch', columns=['ids', 'label_id', 'lens'])
        if split is None:
            return dataset 
        data = dataset.train_test_split(0.2)
        train_data = data["train"]
        valid_data = data["test"]
        return train_data,valid_data
    def collate(self,batch):
        batch_ids = [torch.LongTensor(i['ids']) for i in batch]
    
        ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=self.dictionary["<pad>"], batch_first=True)
        lens = torch.tensor([i['lens'] for i in batch])

        label_ids = torch.tensor([i['label_id'] for i in batch]) 
    
        batch = {'ids': ids,
                'lens': lens,
                'label_id': label_ids}
        return batch

     

if __name__ == "__main__":
    dts = BertDataset("twitter_airline_review/Tweets.csv")
    tr_dt,val_dt = dts.get_dataset()
    # dts = DefaultDataset("twitter_airline_review/Tweets.csv")
    # tr_dt,val_dt = dts.get_dataset()

 
    tr_loader = DataLoader(tr_dt,batch_size=2,shuffle=True,collate_fn=dts.collate)
    for b in tr_loader:
        print(b)
        break 
