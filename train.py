from utils import *
from model import SentimentClassifier,BertSentimentClassifier
from load_data import * 
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm 
import numpy as np 
import torch 
import argparse,logging 
device = "mps"
seed = 0

def add_training_args(parser):
    """
    add training related args 
    """ 
    # data arguments 
    parser.add_argument('--batch-size', default=64, type=int, help='maximum number of sentences in a batch')
        # Add optimization arguments
    parser.add_argument('--max-epoch', default=15, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=4.0, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--patience', default=3, type=int,
                        help='number of epochs without improvement on validation set before early stopping')
    # Add checkpoint arguments
    parser.add_argument('--log-file', default=None, help='path to save logs')
    parser.add_argument('--save-dir',default="checkpoints", help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')
    # model hyperparameters 
    parser.add_argument("--arch",choices=["lstm","transformer","bert"],default="transformer",help="specify the architecture of the model")
    parser.add_argument('--use-feed-forward',default=True ,type=bool, metavar='N', help='num encoder attention heads')
    parser.add_argument("--embed-path",help="specify the path of pre-trained embeddings")
    parser.add_argument('--dropout', type=float,default=0.1,metavar='D', help='dropout probability')
    parser.add_argument('--embed-dim',default=128 ,type=int, metavar='N', help='encoder embedding dimension')
    parser.add_argument('--ffn-embed-dim',default=256 ,type=int, metavar='N', help='encoder embedding dimension for FFN')
    parser.add_argument('--layers', type=int,default=1 ,metavar='N', help='num encoder/lstm layers')
    parser.add_argument('--no-embed-scale',type=bool,default=True,help="specify whether to use embedding scaling")
    parser.add_argument('--num-heads',default=2 ,type=int, metavar='N', help='num encoder attention heads')
    #bert-related hyperparameters
    parser.add_argument('--frozen-layers',default=10,type=int, metavar='N', help='number of encoder layers in bert whose parameters to be frozen')

def get_args():
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    args = parser.parse_args()
  
    return args
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
def get_accuracy(out, label):
    batch_size, _ = out.shape
    _,predicted_classes = out.max(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy
def validate(args,model,criterion, valid_dataset,epoch,collate_fn):
    """ Validates model performance on a held-out development set. """
    valid_loader = \
        DataLoader(valid_dataset, collate_fn=collate_fn,batch_size=args.batch_size)
    progress_bar = tqdm(valid_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)
    model.eval()
    stats = OrderedDict()
    stats['loss'] = 0
    stats['acc'] = 0 
    
    # Iterate over the validation set
    for i, sample in enumerate(progress_bar):
        label = sample["label_id"].to(device) 
        tokens = sample["ids"].to(device)
        lens = sample["lens"].to(device)
        
         
        with torch.no_grad():
            # Compute loss
            out = model(tokens,lens)
            loss = criterion(out,label)
            acc = get_accuracy(out,label)
        
        # Update tracked statistics
        stats['loss'] += loss.item()
        stats['acc'] += acc
  

    # Calculate validation perplexity
    stats['loss'] = stats['loss'] / (i+1)
  
    stats['acc'] =  stats['acc'] / (i+1)


    logging.info('validation: Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(value) for key, value in stats.items()))) 
    acc = stats['acc']
    loss = stats['loss']
    return loss,acc

def train(args,model, criterion,optimizer,train_dataset,epoch,collate_fn):
    model.train()
  
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    stats = OrderedDict()
    stats['loss'] = 0
    stats["acc"] = 0
    progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)
    for i,sample in enumerate(progress_bar):
        
        model.train()
        
        label_id = sample["label_id"].to(device) 
        tokens = sample["ids"].to(device)
        lens = sample["lens"].to(device)
       
        out = model(tokens,lens)
        loss = criterion(out,label_id)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc = get_accuracy(out,label_id)
        loss = loss.item()
        stats['loss'] += loss 
        stats['acc'] += acc 
        progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
                                    refresh=True)

    logging.info('training: Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
        value / len(progress_bar)) for key, value in stats.items())))

def main(args):
    
    torch.manual_seed(seed)
    init_logging(args=args)
    logging.info("zaczyna sie training...")
    logging.info("seed: "+str(seed))


    if args.arch == "bert":
        dts = BertDataset("twitter_airline_review/Tweets.csv")
        train_dataset,valid_dataset = dts.get_dataset()
        collate = dts.collate
        model = BertSentimentClassifier(args=args,pad_id=dts.tokenizer.pad_token_id).to(device=device)
        criterion = nn.CrossEntropyLoss(ignore_index=dts.tokenizer.pad_token_id)
    else:
        dts = DefaultDataset("twitter_airline_review/Tweets.csv")
        dictionary = dts.dictionary
        train_dataset,valid_dataset = dts.get_dataset()
        collate = dts.collate
        model = SentimentClassifier(args,dictionary).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id)

    logging.info('Built a model with {:d} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    state_dict = load_checkpoint(args,model,optimizer)
    last_epoch = state_dict['epoch'] if state_dict is not None else -1
    losses = [] if state_dict is None else state_dict["val_losses"]
    accs = [] if state_dict is None else state_dict["val_accs"]
    best_acc = 0 if state_dict is None else state_dict["best_acc"]
    bad_epochs = 0
    
    for epoch in range(last_epoch + 1, args.max_epoch):
        train(args,model,criterion,optimizer,train_dataset,epoch,collate)
        loss,acc = validate(args, model, criterion, valid_dataset,epoch,collate)
        model.train()
        losses.append(loss)
        accs.append(accs)
        if epoch % args.save_interval == 0:
            save_checkpoint(args=args,model=model,optimizer=optimizer,losses=losses,accs=accs,epoch=epoch,save_file="checkpoint_last.pt",best_acc=best_acc)  # lr_scheduler
        if acc > best_acc:
            save_checkpoint(args=args,model=model,optimizer=optimizer,losses=losses,accs=accs,epoch=epoch,save_file="checkpoint_best.pt",best_acc=best_acc)  # 
            bad_epochs = 0 
            best_acc = acc 
        else:
            bad_epochs += 1 
    if bad_epochs == args.patience:
        logging.info("no improvement for {} batches, training stops!".format(args.patience))

    
if __name__ == "__main__":
    args = get_args()
    main(args)




 




