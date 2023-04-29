import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse 
from utils import get_embedding
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from transformers import BertModel,AutoTokenizer
def add_training_args(parser):
    """
    add training related args 
    """ 
    # model arguments
    parser.add_argument("--arch",choices=["lstm","transformer","bert"],default="lstm",help="specify the architecture of the model")
    # data arguments 
    parser.add_argument("--data",default="stanford_tree_bank",help="the path of the dataset")
    parser.add_argument('--batch-size', default=32, type=int, help='maximum number of sentences in a batch')
        # Add optimization arguments
    parser.add_argument('--max-epoch', default=100, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=4.0, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--patience', default=10, type=int,
                        help='number of epochs without improvement on validation set before early stopping')
    # Add checkpoint arguments
    parser.add_argument('--log-file', default=None, help='path to save logs')
    parser.add_argument('--save-dir', default='checkpoints', help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')


    parser.add_argument("--embed-path",help="specify the path of pre-trained embeddings")
    parser.add_argument('--dropout', type=float,default=0 ,metavar='D', help='dropout probability')
    parser.add_argument('--embed-dim',default=10 ,type=int, metavar='N', help='encoder embedding dimension')
    parser.add_argument('--ffn-embed-dim',default=128 ,type=int, metavar='N', help='encoder embedding dimension for FFN')
    parser.add_argument('--layers', type=int,default=1 ,metavar='N', help='num encoder layers')
    parser.add_argument('--no-embed-scale',type=bool,default=True,help="specify whether to use embedding scaling")
    parser.add_argument('--use-feed-forward',default=True ,type=bool, metavar='N', help='num encoder attention heads')
    parser.add_argument('--num-heads',default=2 ,type=int, metavar='N', help='num encoder attention heads')

def get_args():
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    args = parser.parse_args() 
    return args

def generate_linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

class BertSentimentClassifier(nn.Module):
    def __init__(self,args,pad_id) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        frozen_modules = [self.bert.embeddings,self.bert.encoder.layer[:args.frozen_layers]]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False
        self.proj = nn.Linear(768,3)

        self.do = args.dropout
    def forward(self,tokens,lens):
        mask = (tokens!=self.pad_id).long()
        bert_out = self.bert(input_ids=tokens,
                            attention_mask=mask)[1]
        out = F.dropout(bert_out,p=self.do,training=self.training)
        out = self.proj(out)
        out = F.softmax(out,dim=-1)
        return out


class SentimentClassifier(nn.Module):
    def __init__(self,args,dictionary) -> None:
        super().__init__()
        self.embeddings,embed_dim = get_embedding(dictionary=dictionary,embed_path=args.embed_path,dim=args.embed_dim)
        if embed_dim is not None: #reset embed_dim if we use pre-trained word embeddings 
            args.embed_dim = embed_dim
    
        self.do = args.dropout
        self.pad_id = dictionary["<pad>"] 
        self.arch = args.arch
        if self.arch == "transformer":
            self.encoder = TransformerEncoder(args,dictionary=dictionary)
        if self.arch == "lstm":
            self.encoder = LSTMEncoder(args,dictionary=dictionary)
        self.final_proj = nn.Linear(args.embed_dim,3)
    def forward(self,tokens,lens):
        embeddings = self.embeddings(tokens)
        embeddings = embeddings.contiguous().transpose(0,1)
        if self.arch == "transformer":
            state = self.encoder(embeddings,tokens,lens)
            state = F.dropout(state,p=self.do,training=self.training)
            padding_mask = (~tokens.eq(self.pad_id).unsqueeze(-1))
            state = state * padding_mask        
            state = state.mean(dim=1) / lens.unsqueeze(-1)
        elif self.arch == "lstm":
            state = self.encoder(embeddings,lens)
            state = F.dropout(state,p=self.do,training=self.training)
    
        out = self.final_proj(state)   
        out = F.softmax(out,dim=-1)
       
        return out 

class LSTMEncoder(nn.Module):
    def __init__(self,args,dictionary) -> None:
        super().__init__()
        self.do = args.dropout
        self.layer = args.layers
        self.embeddings = get_embedding(dictionary,dim=args.embed_dim)
        self.projection = nn.Linear(2*args.embed_dim,args.embed_dim)
        self.lstm = nn.LSTM(input_size=args.embed_dim,
                            hidden_size=args.embed_dim,
                            num_layers=self.layer,
                            dropout = self.do,
                            bidirectional=True)

    def forward(self,embeddings,lens):
        packed_state = pack_padded_sequence(embeddings,lengths=lens.data.tolist(),enforce_sorted=False)
        state,(h,c) = self.lstm(packed_state)
        state,lens = pad_packed_sequence(packed_state)

        h = torch.concat([t for t in h[:2]],dim=-1)
        out = self.projection(h)
        out = F.relu(out)

        return out

class TransformerEncoder(nn.Module):
    def __init__(self,args,dictionary) -> None:
        super().__init__()
        self.do = args.dropout 
        self.emb_dim = args.embed_dim 
        self.pad_id = dictionary["<pad>"]
        self.emb_scale = 1.0 if args.no_embed_scale else math.sqrt(self.embed_dim)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args) for _ in range(args.layers) 
        ])

    def forward(self,embeddings,tokens,lens):
       
        forward_state = F.dropout(embeddings,p=self.do,training=self.training)
        padding_mask = tokens.eq(self.pad_id)
        for layer in self.layers:
            forward_state = layer(forward_state,padding_mask=padding_mask)
        forward_state = forward_state.contiguous().transpose(0,1)

        return forward_state 


class TransformerEncoderLayer(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()

       
        self.attn = nn.MultiheadAttention(embed_dim=args.embed_dim,num_heads=args.num_heads,dropout=args.dropout)
            # self.attn = MultiHeadAttention(embed_dim=args.embed_dim,num_attn_heads=args.num_heads)
        self.layernorm1 = nn.LayerNorm(args.embed_dim)
        self.do = args.dropout 
        self.ffn1 = generate_linear(args.embed_dim,args.ffn_embed_dim)
        self.ffn2 = generate_linear(args.ffn_embed_dim,args.embed_dim)
        self.layernorm2 = nn.LayerNorm(args.embed_dim)

        self.ffw = args.use_feed_forward
    def forward(self,state,padding_mask):
        residual = state.clone()
        state,weights = self.attn(state,state,state,key_padding_mask=padding_mask,need_weights=True)
        # turn padded positions to zero
        state = F.dropout(state, p=self.do, training=self.training)
        state += residual
        state = self.layernorm1(state)
 
        residual = state.clone()
        state = self.ffn1(state)
        state = F.dropout(state, p=self.do, training=self.training)
        state = self.ffn2(state)
        state = F.dropout(state, p=self.do, training=self.training)
        state += residual
        state = F.relu(state)
        state = self.layernorm2(state)
        
        return state 

if __name__ == "__main__":
    ids = torch.tensor([[101,1,4,5,1,4],[101,4,5,123,89,10]])
    model = BertSentimentClassifier()
    print(model(ids)) 
 


