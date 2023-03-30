import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse 
from utils import Dictionary,get_embedding
def generate_linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
class Sentiment_Classifier(nn.Module):
    def __init__(self,args,dictionary,parser,label=2) -> None:
        super().__init__()
        self.do = args.dropout
        self.pad_id = dictionary.pad_id  
        self.label = label 
        self.encoder = TransformerEncoder(args,dictionary=dictionary,parser=parser)
        self.final_proj = nn.Linear(args.embed_dim,label)
    def forward(self,tokens):
        state = self.encoder(tokens)
        state = F.dropout(state,p=self.do,training=self.training)
        
        lens = (tokens != self.pad_id).sum(dim=-1).unsqueeze(-1)
        seq_representation = state.mean(dim=1) / lens 
        print(seq_representation.size())
        out = F.softmax(self.final_proj(seq_representation))
        return out 
       
        




        
class TransformerEncoder(nn.Module):
    def __init__(self,args,dictionary:Dictionary,parser) -> None:
        super().__init__()
        args = self._add_args(parser=parser,args=args)

        self.do = args.dropout 
        self.emb_dim = args.embed_dim 
        self.pad_id = dictionary.pad_id 
        self.embeddings = get_embedding(dictionary,embed_path=args.embed_path,dim=self.emb_dim)
        self.emb_scale = 1.0 if args.no_embed_scale else math.sqrt(self.embed_dim)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args) for _ in range(args.layers) 
        ])
    def _add_args(self,parser,args):
        parser.add_argument('--ffn-embed-dim',default=128 ,type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--layers', type=int,default=1 ,metavar='N', help='num encoder layers')
        parser.add_argument('--no-embed-scale',type=bool,default=True,help="specify whether to use embedding scaling")
        if args.arch == "transformer":
            parser.add_argument('--num-heads',default=2 ,type=int, metavar='N', help='num encoder attention heads')
        elif args.arch == "lstm":
            parser.add_argument("--bidirection",type=bool,default=True,help="specify the direction of lstm")
            parser.add_argument("--lstm-layers",type=int,default=1,help="number of lstm layers in the encoder")
        return parser.parse_args()
    def forward(self,tokens):
        embeddings = self.embeddings(tokens)
        forward_state = F.dropout(embeddings,p=self.do,training=self.training)
        forward_state = forward_state.transpose(0,1)
        padding_mask = tokens.eq(self.pad_id)
        for layer in self.layers:
            forward_state = layer(forward_state,padding_mask=padding_mask)
        forward_state = forward_state.transpose(0,1)
        return forward_state 

class TransformerEncoderLayer(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.arch = args.arch
        if self.arch == "transformer":
            self.attn = MultiHeadAttention(embed_dim=args.embed_dim,num_attn_heads=args.num_heads)
        else:
            direction = 2 if args.bidirection else 1
            self.lstm = nn.LSTM(input_size=args.embed_dim,hidden_size=args.embed_dim,bidirectional=args.bidirection)
            self.merge_direction = generate_linear(args.embed_dim*direction,args.embed_dim)
        self.layernorm1 = nn.LayerNorm(args.embed_dim)
        self.do = args.dropout 
        self.ffn1 = generate_linear(args.embed_dim,args.ffn_embed_dim)
        self.ffn2 = generate_linear(args.ffn_embed_dim,args.embed_dim)
        self.layernorm2 = nn.LayerNorm(args.embed_dim)
    def forward(self,state,padding_mask):
        residual = state.clone()
        if self.arch == "transformer":
            state,_ = self.attn(state,state,state,key_padding_mask=padding_mask) 
        else:
            state,(h,c) = self.lstm(state)
            state = self.merge_direction(state)
        # turn padded positions to zero
        padding_mask = padding_mask.transpose(0,1).unsqueeze(-1)
        state = state.masked_fill_(padding_mask,0)

        state = F.dropout(state,p=self.do,training=self.training)
        state += residual
        state = self.layernorm1(state)

        residual = state.clone()
        state = F.relu(self.ffn1(state))
        state = F.dropout(state,p=self.do,training=self.training)
        state = self.ffn2(state)
        state += residual 
        state = self.layernorm2(state)
        return state 

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self,
                 embed_dim,
                 num_attn_heads,
                 kdim=None,
                 vdim=None,
                 dropout=0.25,
                 encoder_decoder_attention=False):
        '''
        ___QUESTION-6-MULTIHEAD-ATTENTION-NOTE
        You shouldn't need to change the __init__ of this class for your attention implementation
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = vdim if vdim else embed_dim
        self.num_heads = num_attn_heads
        self.attention_dropout = dropout
        self.head_embed_size = embed_dim // num_attn_heads
        self.head_scaling = math.sqrt(self.head_embed_size)

        self.enc_dec_attention = encoder_decoder_attention

        kv_same_dim = self.k_embed_size == embed_dim and self.v_embed_size == embed_dim
      
        assert self.head_embed_size * self.num_heads == self.embed_dim, "Embed dim must be divisible by num_heads!"
        self.k_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.v_proj = nn.Linear(self.v_embed_size, embed_dim, bias=True)
        self.q_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        # Xavier initialisation
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                attn_mask=None,
                need_weights=True):

        # Get size features
        tgt_time_steps, batch_size, embed_dim = query.size()
        assert self.embed_dim == embed_dim

        '''
        ___QUESTION-6-MULTIHEAD-ATTENTION-START
        Implement Multi-Head attention  according to Section 3.2.2 of https://arxiv.org/pdf/1706.03762.pdf.
        Note that you will have to handle edge cases for best model performance. Consider what behaviour should
        be expected if attn_mask or key_padding_mask are given?
        '''

        # attn is the output of MultiHead(Q,K,V) in Vaswani et al. 2017
        # attn must be size [tgt_time_steps, batch_size, embed_dim]
        # attn_weights is the combined output of h parallel heads of Attention(Q,K,V) in Vaswani et al. 2017
        # attn_weights must be size [num_heads, batch_size, tgt_time_steps, key.size(0)]
        # TODO: REPLACE THESE LINES WITH YOUR IMPLEMENTATION ------------------------ CUT

        # some ideas of implementation come from: Transformer from Scratch, Peter Bloem (2019)
        # use simpler variables: 
        b,kt,qt,h,hd = batch_size,key.size(0),tgt_time_steps,self.num_heads,self.head_embed_size 

        attn = torch.zeros(size=(qt, b, embed_dim))
        attn_weights = torch.zeros(size=(h, b, qt, kt)) if need_weights else None

        # q,k,v: [time_steps, batch_size, num_heads, head_embed_size]
        q = self.q_proj(query).view(qt,b,h,hd) 
        k = self.k_proj(key).view(kt,b,h,hd)
        v = self.v_proj(value).view(kt,b,h,hd)

        # q,k,v: [batch_size * num_heads, time_steps, head_embed_size]
        q = q.view(qt,b*h,hd).transpose(0,1).contiguous()
        k = k.view(kt,b*h,hd).transpose(0,1).contiguous() 
        v = v.view(kt,b*h,hd).transpose(0,1).contiguous()

        # attention score: [batch_size * num_heads, query_time_steps, key_time_steps]
        score = torch.bmm(q,k.transpose(1, 2)) / self.head_scaling
        # attention score: [batch_size * num_heads, query_time_steps, key_time_steps]
        score = score.view(h,b,qt,kt)
        
        if attn_mask is not None:
            score += attn_mask 
        if key_padding_mask is not None: 
            # key_padding_mask: [1, batch_size, 1, key_time_steps]
            key_padding_mask = key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=0)
            score.masked_fill_(key_padding_mask,float("-inf"))
         
        attn_weights = F.softmax(score, dim=-1)
        # out : [batch_size * num_heads, query_time_steps, head_embed_size] 
        out = torch.bmm(attn_weights.view(b*h,qt,kt), v)
        # out： [query_time_steps, ,batch_size, num_heads, head_embed_size]
        out = out.transpose(0,1).contiguous().view(qt,b,h,hd) 
        # attn: [query_time_steps, batch_size, embed_dim]
        out = out.view(qt,b,hd*h)
        attn += self.out_proj(out)
        # TODO: --------------------------------------------------------------------- CUT
        '''
        ___QUESTION-6-MULTIHEAD-ATTENTION-END
        '''
        return attn, attn_weights if need_weights else None







        


if __name__ == "__main__":
    dic_dir = "stanford_tree_bank/data_raw/dictionary.txt"
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--arch",choices=["lstm","transformer"],default="transformer",help="specify the architecture of the model")
        parser.add_argument("--embed-path",help="specify the path of pre-trained embeddings")
        return parser
    def main():
        parser = get_parser()
        args = parser.parse_args()
        dic = Dictionary.load_dict(dic_dir)
        model = TransformerEncoder(args=args,dictionary=dic,parser=parser)
        args = parser.parse_args()

        
    main()
