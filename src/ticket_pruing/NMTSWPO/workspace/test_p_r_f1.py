#encoding=utf-8
import sys

#reload(sys)
#sys.setdefaultencoding('utf-8')

# sys.path.insert(0, '/home/gitProject/dgx1-OpenNMT-py')
# sys.path.insert(0, '/home/gitProject/BLEU4Python')
sys.path.insert(0, '/fl')
sys.path.insert(0, '/fl/NMTSWPO')
sys.path.insert(0, '/fl/LMSWPO/tnnls_workspace')
sys.path.insert(0, '/fl/LMSWPO/tnnls_workspace/package')
sys.path.insert(0, '/fl/NMTSWPO/workspace')
print(sys.path)


import os
import torch
from torch.multiprocessing import Process
if __name__ == "__main__":
  torch.multiprocessing.set_start_method('spawn')

import torch.nn as nn
from torch.autograd import Variable
from torch import cuda
import torch.distributed as dist
from rouge.rouge import Rouge
#import ncs
from easydict import EasyDict as edict
from onmt.Utils import use_gpu
import onmt
import bleu
import math
import time
import ncs
import numpy as np
from logger import Logger
import os,os.path,datetime
from masked_network import MaskedModel
import copy
from layer_group import group_dict #group_dict1, group_dict2
from nlgeval import NLGEval

logger = Logger('/home/test/opennmt-prune-logs')

te = 'NMTSWPO/workspace/opennmt_translate_dummy_opt.pt'
print(os.path.exists(te))
print(os.path.exists('NMTSWPO/workspace/opennmt_translate_opt.pt'))
print(os.path.exists('opennmt_translate_opt.pt'))


#weights='/home/lgy/deepModels/torch_models/opennmt-py/zh_bahdanau_acc_20.51_ppl_338.36_e1.pt'
#TRAIN_DATA  = '/home/lgy/data/wmt/wmt17-en-zh/pywmt17'
TRAIN_DATA  = '/fl/data/wmt/wmt14-de-en/len50_pywmt14'
SAVE_MODEL_PATH = '/fl/deepModels/torch_models/opennmt-py/prune/leiluong'
SAVE_MODEL_FOLDER = '/fl/deepModels/torch_models/opennmt-py/tmp_prune/'
print(os.path.exists('/fl/data'))
print(os.path.exists(TRAIN_DATA))
print(os.path.exists(SAVE_MODEL_PATH))
print(os.path.exists(SAVE_MODEL_FOLDER))
l=os.listdir(SAVE_MODEL_FOLDER)
l.sort(key=lambda fn: os.path.getmtime(SAVE_MODEL_FOLDER+fn) if not os.path.isdir(SAVE_MODEL_FOLDER+fn) else 0)


weights = '/fl/deepModels/SiPO/original_model/RNNSearch/rnnsearch_original.pt'

rnn_path = '/fl/deepModels/SiPO/original_model/RNNSearch/rnnsearch_original.pt'

GPU_ID = 0
other_GPU_IDs = [1] # devices used for individuals, one device id for one individual
REPORT_EVERY = 50
EPOCHS = 1
MAX_GRAD_NORM = 1
LEARNING_RATE = 1.0
START_DECAY_AT = 8
TEST_BATCH_SIZE= 64


#----------------from OpenNMT-py----------------------------------
def load_fields(train, valid, checkpoint, opt):
    fields = onmt.IO.load_fields(
                torch.load(opt.data + '.vocab.pt'))
    fields = dict([(k, f) for (k, f) in fields.items()
                  if k in train.examples[0].__dict__])
    train.fields = fields
    valid.fields = fields

    if opt.train_from:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.IO.load_fields(checkpoint['vocab'])

    print(' * vocabulary size. source = %d; target = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab)))

    return fields

def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model



def translate_opt_initialize(trans_p, trans_dum_p):
   translate_opt = torch.load(trans_p)
   translate_dummy_opt = torch.load(trans_dum_p)
   #   translate
   translate_opt.model = weights
   #   dataset for pruning
   #translate_opt.src = '/home/lgy/data/wmt/wmt17-en-zh/smalltrain-test-en.txt.tok'
   #translate_opt.tgt = '/home/lgy/data/wmt/wmt17-en-zh/smalltrain-test-zh.txt.tok'
   translate_opt.src = '/fl/data/wmt/wmt14-de-en/from_sru/en-de/en-test.txt'
   translate_opt.tgt = '/fl/data/wmt/wmt14-de-en/from_sru/en-de/de-test.txt'
   translate_opt.start_epoch = 2
   translate_opt.model = weights
   translate_opt.gpu = GPU_ID
   #translate_opt.beam_size = 1

   return translate_opt, translate_dummy_opt

def opt_initialize(c_point, trans_p, trans_dum_p):
   train_opt = c_point['opt']

   # here is the custom configuration
   #   train
   train_opt.gpuid = [GPU_ID]
   train_opt.start_epoch = train_opt.epochs + 1
   train_opt.epochs = EPOCHS
   train_opt.max_grad_norm = MAX_GRAD_NORM
   train_opt.learning_rate = LEARNING_RATE
   train_opt.start_decay_at = START_DECAY_AT
   train_opt.train_from = weights
   train_opt.data = TRAIN_DATA 
   train_opt.save_model = SAVE_MODEL_PATH

   translate_opt, translate_dummy_opt = translate_opt_initialize(trans_p, trans_dum_p)

   return train_opt, translate_opt, translate_dummy_opt

def init_train_model(c_point, opt, fields):
    model_opt = c_point['opt']
    model_opt.gpuid = opt.gpuid
    return build_model(model_opt, opt, fields, c_point)

def init_translate_model(opt, dummy_opt):
    return onmt.Translator(opt, dummy_opt.__dict__)


#   Set the crates of each layer, the pruning will happen in the next forward action

def apply_MP_on_mask(thresholds, mask_dict, orig_dict, layer2group_map_dict, sorted_group_parameters, group_name_list):
    assert len(thresholds) == len(group_name_list)

    threshold_dict = {}
    for i in range(len(group_name_list)):
       group_name = group_name_list[i]
       tmp_threshold_ratio = 1. - thresholds[i]
       _indx = tmp_threshold_ratio * (sorted_group_parameters[group_name].nelement()-1) 
       threshold_dict[group_name] =  sorted_group_parameters[group_name][int(_indx)]
       
    for the_name, the_param in mask_dict.items():
      group_name = layer2group_map_dict[the_name]
      tmp_v = threshold_dict[group_name]
      tmp_size = the_param.size()
      tmp_m = Variable(orig_dict[the_name].data.new(tmp_size).fill_(tmp_v))
      the_param.data.copy_((orig_dict[the_name].gt(tmp_m) + orig_dict[the_name].lt(tmp_m.neg())).data)

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

def save_txt(path, lis):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lis:
            f.write(line + '\n')

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
  nlg_ref = [[x[0] for x in references ]]
  
  nlg_eval = NLGEval()
  
  metrics_eval = nlg_eval.compute_metrics(nlg_ref, hypothesis)
  print(metrics_eval)
  print('BLEU: {}'.format(bleu_score))
  # training/validation 阶段的ppl计算在onmt/Trainer.py的Statisci()中；translating的ppl计算在 translate.py中的reprot_score函数里
  print('PPL: {}'.format(ppl))

  return torch.FloatTensor([ppl, bleu_score, 0.0])# the last reserved for rank number



#------mian--------------
def main():

  
  valid_data = torch.load(TRAIN_DATA + '.valid.pt')
  fields = onmt.IO.load_fields(torch.load(TRAIN_DATA + '.vocab.pt'))
  valid_data.fields = fields # we need to clear this assignment relationg if we want to transfere valid among threads

  if GPU_ID == 0 or GPU_ID == 1:
    cuda.set_device(GPU_ID)
    with cuda.device(GPU_ID):
        # '/fl/deepModels/tmp/rnnsearch_tmp.pt','/fl/deepModels/tmp/loungnet_tmp.pt'
        checkpoint_path = '/fl/deepModels/tmp/loungnet_tmp.pt'
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        ref_model = onmt.ModelConstructor.make_base_model(model_opt, fields, True, checkpoint)
        ref_model.eval()
        ref_model.generator.eval()
        masked_model = MaskedModel(ref_model, group_dict, cuda.current_device(), cuda.current_device()) # ref_model is at current_device, no copy will happen
         # train data loading
  
        
        
    
    translate_opt, translate_dummy_opt = translate_opt_initialize('/fl/NMTSWPO/workspace/opennmt_translate_opt.pt', '/fl/NMTSWPO/workspace/opennmt_translate_dummy_opt.pt')
    translator = init_translate_model(translate_opt, translate_dummy_opt)
    del translator.model
    translator.model = masked_model
    tt=open(translate_opt.tgt, 'r', encoding='utf-8')
    references = [[t] for t in tt]

    p = 0.43
    translate_data = onmt.IO.ONMTDataset(
    translate_opt.src, translate_opt.tgt, fields,
    use_filter_pred=False)
    prune_data = onmt.IO.OrderedIterator(
    dataset=translate_data, device=GPU_ID,
    batch_size=1, train=False, sort=False,
    shuffle=False)
    tmp_crate = len(masked_model.group_name_list)*[p]
    
    masked_model.change_mask(tmp_crate, apply_MP_on_mask)
    masked_model.apply_mask()
    
    tmp_fit = evaluate_trans(translator, references, prune_data, translate_data)
    logger.scalar_summary('test_bleu', tmp_fit[1]*100, int(p*100))
    
    logger.scalar_summary('test_ppl', tmp_fit[0], int(p*100))
    
    print('percentage %s => bleu (%.4f), ppl (%.4f)' % (p*100, tmp_fit[1]*100, tmp_fit[0]))
      
      
    
    
    
if __name__ == "__main__":
  #torch.multiprocessing.set_start_method('spawn')
  main()
