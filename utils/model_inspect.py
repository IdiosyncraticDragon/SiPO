import sys
import model
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import masked_networkLM
import data

def lm_data_load(data_path, batch_size = 10):
   corpus = data.Corpus(data_path)
   train_data = batchify(corpus.train, batch_size)
   val_data = batchify(corpus.valid, batch_size)
   test_data = batchify(corpus.test, batch_size)
   return corpus, train_data, val_data, test_data

def lm_model_load(model_path):
   ref_m = None
   with open(model_path,'rb') as ff:
      ref_m = torch.load(ff)
   return ref_m

def lm_original_param_num(ref_m):
   aaa = [x.numel() for a,x in ref_m.named_parameters()]
   return np.sum(aaa)

def batchify(data, bsz, is_cuda=True):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if is_cuda:
        data = data.cuda()
    return data

def get_batch(source, i, evaluation=False, bptt=35):
    seq_len = min(bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(model, data_source, corpus, batch_size = 10, bptt = 35):
    # Turn on evaluation mode which disables dropout.
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    total_acc = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        #pdb.set_trace()
        output_flat = output.view(-1, ntokens)
        _, preds = torch.max(output_flat, 1)
        total_acc += (preds == targets).data.cpu().sum()
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source), total_acc*1. / data_source.nelement()


def train(model, batch_size = 10, bptt=35, clip=0.25, log_interval=200):
    # Turn on training mode which enables dropout.
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def set_env(seed, is_cuda=True):
   torch.manual_seed(seed)
   if torch.cuda.is_available():
       if not is_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
       else:
           torch.cuda.manual_seed(seed)

def lm_evaluate(model, corpus, _data, batch_size = 10, is_cuda=True):
   ntokens = len(corpus.dictionary)
   if is_cuda:
      model.cuda()
   loss, acc = evaluate(model, _data, corpus)
   return math.exp(loss), acc
