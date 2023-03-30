import torch 
import torch.nn as nn 
import logging
class Dictionary(object):
    def __init__(self,pad="<pad>",unk="<unk>") -> None:
        self.pad = pad 
        self.unk = unk 
        self.pad_id = 0 
        self.unk_id = 1
        self.word2id = {self.pad:0,self.unk:1}
        self.words = []
        self.counts = []
    def __len__(self):
        return len(self.words)
    def __getitem__(self,index):
        return self.words[index] if index < len(self.word2id) else self.unk 
    @classmethod
    def load_dict(cls,dict_dir:str):
        """
        load a dictionary from the dictionary file

        return: dictionary object 
        """
        text_file = open(dict_dir,"r").readlines()
        dictionary = cls()
        for i,tp in enumerate(text_file):
            w = tp.split(" ")[0]
            dictionary.word2id[w] = i+2 
            dictionary.words.append(w)
        return dictionary  
    def binarize_sents(self,sents:list):
        """
        Convert a list of sent to indeces 

        return torch.tensor
        """
        word2id = self.word2idx
        max_len = max([len(s.split()) for s in sents])
        ids = [[word2id.get(w,word2id[self.unk]) for w in s.split()] for s in sents]
        ids = torch.tensor([s+[self.padid]*(max_len-len(s)) for s in ids],dtype=torch.int64) 
        return ids


def get_embedding(dictionary,embed_path=None,dim=None):
    """Parse an embedding text file into an torch.nn.Embedding layer."""
    embed_dict, embed_dim = {}, None
    if embed_path is not None:
        with open(embed_path) as file:
            embed_dim = int(next(file).rstrip().split(" ")[1])
            for line in file:
                tokens = line.rstrip().split(" ")
                embed_dict[tokens[0]] = torch.Tensor([float(weight) for weight in tokens[1:]])
        embedding = nn.Embedding(len(dictionary)+2, embed_dim, dictionary.pad_id)
        #the dictionary doesn't include unk and pad, thus the embedding length must add 2. 
        embedding.weight.data[0] = torch.rand(embed_dim) #the embedding for pad
        embedding.weight.data[1] = torch.rand(embed_dim) #the embedding for unk
        for idx, word in enumerate(dictionary):
            idx = idx+2 
            if word in embed_dict:
                embedding.weight.data[idx] = embed_dict[word]
            else:
                embedding.weight.data[idx] = torch.rand(embed_dim)
    else:
        logging.info("randomly initialize word embeddings")
        assert dim is not None 
        embedding = nn.Embedding(len(dictionary),dim,dictionary.pad_id)
    return embedding
def test():
    return 123

if __name__ == "__main__":
    dir = "stanford_tree_bank/data_raw/dictionary.txt"
    dic = Dictionary.load_dict(dir)
   