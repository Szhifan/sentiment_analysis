from utils import *
from transformer import Sentiment_Classifier
import argparse
dic_dir = "stanford_tree_bank/data_raw/dictionary.txt"
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",choices=["lstm","transformer"],default="transformer",help="specify the architecture of the model")
    parser.add_argument("--embed-path",help="specify the path of pre-trained embeddings")
    parser.add_argument('--dropout', type=float,default=0.1 ,metavar='D', help='dropout probability')
    parser.add_argument('--embed-dim',default=100 ,type=int, metavar='N', help='encoder embedding dimension')


    return parser
def main():
    data = torch.tensor([[1,2,3,4,5],[8,9,6,4,0],[4,5,6,0,0]])
    parser = get_parser()
    args = parser.parse_args()
    dic = Dictionary.load_dict(dic_dir)
    model = Sentiment_Classifier(args=args,dictionary=dic,parser=parser)
    args = parser.parse_args()
    out = model(data)
 




main()