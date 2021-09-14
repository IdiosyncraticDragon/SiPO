import sys
sys.path.insert(0, '../')
import model
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import cuda
import time
import math
import masked_networkLM
from masked_network import MaskedModel
from onmt.Utils import use_gpu
import onmt
import data
import bleu

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

def lm_evaluate(model, corpus, data_source, batch_size = 10, is_cuda=True, bptt=35):
   criterion = nn.CrossEntropyLoss()
   model.eval()
   total_loss = 0
   total_acc = 0
   ntokens = len(corpus.dictionary)
   if is_cuda:
      model.cuda()
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
   acc = total_acc*100. / data_source.nelement()
   loss = total_loss[0] / len(data_source)
   #loss, acc = evaluate(model, _data, corpus)
   return math.exp(loss), acc

def translate_opt_initialize(trans_p, trans_dum_p, data_path, model_path, GPU_ID):
   translate_opt = torch.load(trans_p)
   translate_dummy_opt = torch.load(trans_dum_p)
   #   translate
   translate_opt.model = model_path
   #   dataset for pruning
   translate_opt.src = '{}/en-test.txt'.format(data_path)
   translate_opt.tgt = '{}/de-test.txt'.format(data_path)
   translate_opt.start_epoch = 2
   translate_opt.model = model_path
   translate_opt.gpu = GPU_ID
   return translate_opt, translate_dummy_opt

def init_translate_model(opt, dummy_opt):
    return onmt.Translator(opt, dummy_opt.__dict__)

#  evaluate the accuracy of a network with a set of crates respect to a original accuracy
class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, loss=0., n_words=0., n_correct=0.):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

def make_valid_data_iter(valid_data, batch_size, gpu_id=None):
    """
    This returns user-defined validate data iterator for the trainer
    to iterate over during each validate epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    is ok too.
    """
    return onmt.IO.OrderedIterator(
                dataset=valid_data, batch_size= batch_size,
                device= gpu_id if gpu_id is not None else GPU_ID,
                train=False)

def make_loss_compute(model, tgt_vocab, dataset, gpu_id=None, copy_attn=False, copy_attn_force=False):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, dataset, copy_attn_force)
    else:
        compute = onmt.Loss.NMTLossCompute(model.generator, tgt_vocab)

    if gpu_id == None:
      gpu_id = cuda.current_device()
    compute.cuda(gpu_id)

    return compute

def evaluate(thenet, valid_data, fields, batch_size = 64, gpu_id=None):
   gpu_used = gpu_id if gpu_id is not None else torch.cuda.current_device()
   valid_iter = make_valid_data_iter(valid_data, batch_size, gpu_used)
   valid_loss = make_loss_compute(thenet, fields["tgt"].vocab, valid_data, gpu_used)

   stats = Statistics()

   for batch in valid_iter:
      _, src_lengths = batch.src
      src = onmt.IO.make_features(batch, 'src')
      tgt = onmt.IO.make_features(batch, 'tgt')

      # F-prop through the model.
      outputs, attns, _ = thenet(src, tgt, src_lengths)

      # Compute loss.
      batch_stats = valid_loss.monolithic_compute_loss(
      batch, outputs, attns)
      # Update statistics.
      stats.update(batch_stats)

   return torch.FloatTensor([stats.ppl(), stats.accuracy()])# the last two 0.0 reserved for rank number, and sparsity

def evaluate_trans(thenet, references, vali_data, vali_raw_data):
  hypothesis = []
  score_total = 0.
  num_word_total = 0
  for batch in vali_data:
     pred_batch, gold_batch, pred_scores, gold_scores, attn, src = thenet.translate(batch, vali_raw_data)
     score_total += sum([score[0] for score in pred_scores])
     num_word_total += sum(len(x) for x in batch.tgt[1:]) 
     hypothesis.extend([' '.join(x[0]) for x in pred_batch])
  ppl = math.exp(-score_total/num_word_total)
  bleu_score = bleu.corpus_bleu(hypothesis, references)[0][0] #[final, n-gram1,n-gram2,...], [bp, ...]
  #print('BLEU: {}'.format(bleu_score))
  # training/validation 阶段的ppl计算在onmt/Trainer.py的Statisci()中；translating的ppl计算在 translate.py中的reprot_score函数里
  #print('PPL: {}'.format(ppl))

  return torch.FloatTensor([ppl, bleu_score, 0.0])# the last reserved for rank number

def nmt_test(model_path_pruned, DATA_PATH, GPU_ID, translate_param1_path, translate_param2_path, group_dict):
     cuda.set_device(GPU_ID)
     valid_data = torch.load(DATA_PATH + 'len50_pywmt14.valid.pt')
     fields = onmt.IO.load_fields(torch.load(DATA_PATH + 'len50_pywmt14.vocab.pt'))
     valid_data.fields = fields 
     checkpoint = torch.load(model_path_pruned, map_location=lambda storage, loc: storage)
     model_opt = checkpoint['opt']
     with cuda.device(GPU_ID):
         ref_model = onmt.ModelConstructor.make_base_model(model_opt, fields, True, checkpoint)
         ref_model.eval()
         ref_model.generator.eval()
         masked_model = MaskedModel(ref_model, group_dict, cuda.current_device(), cuda.current_device()) # ref_model is at current_device, no copy will happen
     translate_opt, translate_dummy_opt = translate_opt_initialize(translate_param1_path, translate_param2_path, DATA_PATH, model_path_pruned, GPU_ID)
     translator = init_translate_model(translate_opt, translate_dummy_opt)
     del translator.model
     translator.model = ref_model
     tt=open(translate_opt.tgt, 'r')
     references = [[t] for t in tt]

     translate_data = onmt.IO.ONMTDataset(
       translate_opt.src, translate_opt.tgt, fields,
       use_filter_pred=False)
     prune_data = onmt.IO.OrderedIterator(
       dataset=translate_data, device=GPU_ID,
       batch_size=1, train=False, sort=False,
       shuffle=False)

     sparsity = masked_model.get_sparsity()
     total_param = masked_model.total_parameters_of_pretrain()

     tmp_fit1 = evaluate(masked_model, valid_data, fields)
     tmp_fit2 = evaluate_trans(translator, references, prune_data, translate_data)
     return total_param, sparsity, tmp_fit1, tmp_fit2
