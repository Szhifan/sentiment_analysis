import torch 
import torch.nn as nn 
import logging
import os 
import sys 
from torch.serialization import default_restore_location



def get_embedding(dictionary,embed_path=None,dim=None):
    """Parse an embedding text file into an torch.nn.Embedding layer."""
    embed_dict, embed_dim = {}, None
    if embed_path is not None:
        with open(embed_path) as file:
            embed_dim = int(next(file).rstrip().split(" ")[1])
            for line in file:
                tokens = line.rstrip().split(" ")
                embed_dict[tokens[0]] = torch.Tensor([float(weight) for weight in tokens[1:]])
        embedding = nn.Embedding(len(dictionary), embed_dim, dictionary["<pad>"])
        #the dictionary doesn't include unk and pad, thus the embedding length must add 2. 
        vocab_list = dictionary.get_itos()
        for idx, word in enumerate(vocab_list):
            if word in embed_dict:
                embedding.weight.data[idx] = embed_dict[word]
            else:
                embedding.weight.data[idx] = torch.rand(embed_dim)
    else:
        logging.info("randomly initialize word embeddings")
        assert dim is not None 
        embedding = nn.Embedding(len(dictionary),dim,dictionary["<pad>"])
    return embedding,embed_dim

def init_logging(args):
    handlers = [logging.StreamHandler()]
    if hasattr(args, 'log_file') and args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode='a+'))
    logging.basicConfig(handlers=handlers, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.info('COMMAND: %s' % ' '.join(sys.argv))
    logging.info('Arguments: {}'.format(vars(args)))

def save_checkpoint(args, model, optimizer,best_acc,losses,accs,epoch,save_file):
    if args.save_dir is None:
        return 
    os.makedirs(args.save_dir, exist_ok=True)
    state_dict = {
        'epoch': epoch,
        'val_losses': losses,
        'val_accs':accs,
        "best_acc":best_acc,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args,
    }
    torch.save(state_dict, os.path.join(args.save_dir, save_file))



def load_checkpoint(args, model, optimizer):
    if args.save_dir is None:
        return 
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        print("loading checkpoint from: {}".format(checkpoint_path))
        state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        logging.info('Loaded checkpoint {}'.format(checkpoint_path))
        return state_dict



if __name__ == "__main__":
    dir = "stanford_tree_bank/data_raw/dictionary.txt"

   