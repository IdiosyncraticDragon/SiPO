#encoding=utf-8
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
sys.path.insert(0, '/home/lgy/gitProject/OpenNMT-py')
sys.path.insert(0, '/home/lgy/gitProject/BLEU4Python')
from torch.multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import pdb

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import cuda
import torch.distributed as dist
from torch.multiprocessing import Process
#import ncs
from easydict import EasyDict as edict
from onmt.Utils import use_gpu
import onmt
import bleu
import math
import time
import ncs
import numpy as np

#weights='/home/lgy/deepModels/torch_models/opennmt-py/zh_bahdanau_acc_20.51_ppl_338.36_e1.pt'
weights='/home/lgy/deepModels/torch_models/opennmt-py/leiluong_acc_58.34_ppl_7.51_e12.pt'
#TRAIN_DATA  = '/home/lgy/data/wmt/wmt17-en-zh/pywmt17'
TRAIN_DATA  = '/home/lgy/data/wmt/wmt14-de-en/len50_pywmt14'
GPU_ID = 3
GPU_ID2 = 4
REPORT_EVERY = 50
EPOCHS = 1
MAX_GRAD_NORM = 1
LEARNING_RATE = 1.0
START_DECAY_AT = 8
  
if GPU_ID:
  cuda.set_device(GPU_ID)

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

def build_optim(model, checkpoint, opt):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        # what members of opt does Optim need?
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt
        )

    optim.set_parameters(model.parameters())

    return optim

def make_train_data_iter(train_data, opt):
    """
    This returns user-defined train data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    return onmt.IO.OrderedIterator(
                dataset=train_data, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                repeat=False)

def make_valid_data_iter(valid_data, opt):
    """
    This returns user-defined validate data iterator for the trainer
    to iterate over during each validate epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    is ok too.
    """
    return onmt.IO.OrderedIterator(
                dataset=valid_data, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                train=False, sort=True)

def make_loss_compute(model, tgt_vocab, dataset, opt):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, dataset, opt.copy_attn_force)
    else:
        compute = onmt.Loss.NMTLossCompute(model.generator, tgt_vocab)

    if use_gpu(opt):
        compute.cuda()

    return compute

def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % REPORT_EVERY == -1 % REPORT_EVERY:
        report_stats.output(epoch, batch+1, num_batches, start_time)
        report_stats = onmt.Statistics()

    return report_stats

def train_model(model, train_data, valid_data, fields, optim, opt):

    train_iter = make_train_data_iter(train_data, opt)
    valid_iter = make_valid_data_iter(valid_data, opt)

    train_loss = make_loss_compute(model, fields["tgt"].vocab,
                                   train_data, opt)
    valid_loss = make_loss_compute(model, fields["tgt"].vocab,
                                   valid_data, opt)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches

    trainer = onmt.Trainer(model, train_iter, valid_iter,
                           train_loss, valid_loss, optim,
                           trunc_size, shard_size)
    pdb.set_trace()

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(opt, epoch, fields, valid_stats)
#----------------end OpenNMT-py----------------------------------

def translate_opt_initialize(trans_p, trans_dum_p):
   translate_opt = torch.load(trans_p)
   translate_dummy_opt = torch.load(trans_dum_p)
   #   translate
   translate_opt.model = weights
   #   dataset for pruning
   #translate_opt.src = '/home/lgy/data/wmt/wmt17-en-zh/smalltrain-test-en.txt.tok'
   #translate_opt.tgt = '/home/lgy/data/wmt/wmt17-en-zh/smalltrain-test-zh.txt.tok'
   translate_opt.src = '/home/lgy/data/wmt/wmt14-de-en/from_sru/en-de/en-test.txt'
   translate_opt.tgt = '/home/lgy/data/wmt/wmt14-de-en/from_sru/en-de/de-test.txt'
   translate_opt.start_epoch = 2
   translate_opt.model = weights
   translate_opt.gpu = GPU_ID

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

   translate_opt, translate_dummy_opt = translate_opt_initialize(trans_p, trans_dum_p)

   return train_opt, translate_opt, translate_dummy_opt

def init_train_model(c_point, opt, fields):
    model_opt = c_point['opt']
    model_opt.gpuid = opt.gpuid
    return build_model(model_opt, opt, fields, c_point)

def init_translate_model(opt, dummy_opt):
    return onmt.Translator(opt, dummy_opt.__dict__)

def test_net(thenet, _count=1):
    pass

def sorted_w(the_model):
    module_name_dict = {}
    sort_tensors= {}
    thresholds = []
    count = 0
    for param_name, module_tensor in the_model.named_parameters():
       module_name_dict[param_name] = count
       sort_tensors[param_name],_ = module_tensor.view(-1).sort(descending=True)
       thresholds.append(int(module_tensor.view(-1).size()[0]-1))
       count += 1
    print(module_name_dict)
    print(thresholds)
    return module_name_dict, thresholds, sort_tensors

#   Set the crates of each layer, the pruning will happen in the next forward action
def apply_prune(the_model, ref_model, sort_w, name_dict, thresholds):
    for the_name, the_param in the_model.named_parameters():
      tmp_inds = name_dict[the_name]
      tmp_threshold = thresholds[tmp_inds]
      tmp_v = sort_w[the_name][tmp_threshold]
      tmp_size = the_param.size()
      tmp_m = Variable(torch.cuda.FloatTensor(tmp_size).copy_(tmp_v.data)).cuda() 
      the_param.data.zero_()
      the_param.data = the_param.addcmul(ref_model[the_name].gt(tmp_m).float(), ref_model[the_name]).data
    return the_model

#  calcuate the sparsity of a network model
def get_sparsity(thenet):
   total_num = 0
   pruned_num = 0.
   for module_tensor in thenet.parameters():
     total_num += module_tensor.view(-1).size()[0]
     pruned_num += module_tensor.eq(0).float().sum().data[0]

   return (total_num - pruned_num)/total_num
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

def evaluate(thenet, valid_data, fields, opt):
   valid_iter = make_valid_data_iter(valid_data, opt)
   valid_loss = make_loss_compute(thenet, fields["tgt"].vocab, valid_data, opt)

   thenet.eval()
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

   return torch.FloatTensor([stats.ppl(), stats.accuracy(), 0.0])# the last reserved for rank number

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
  print('BLEU: {}'.format(bleu_score))
  # training/validation 阶段的ppl计算在onmt/Trainer.py的Statisci()中；translating的ppl计算在 translate.py中的reprot_score函数里
  print('PPL: {}'.format(ppl))

  return torch.FloatTensor([ppl, bleu_score, 0.0])# the last reserved for rank number

# multi process
#------------------------------for parallel--------------------
def init_processes(rank, size, param_name, prune_threshold, fn, model_dict, sorted_weights, results, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, param_name, prune_threshold, model_dict, sorted_weights, results)

def prune_and_eval(rank, size, param_name, prune_threshold, ref_model_dict, ref_sorted_weights, results):
   local_ref_model_dict = ref_model_dict
   local_sorted_weights = ref_sorted_weights
   gpu_id = GPU_ID

   if rank>=4 and rank <size: # split tasks to different GPUs
     gpu_id = GPU_ID2
     cuda.set_device(gpu_id)

   local_checkpoint = torch.load(weights, map_location=lambda storage, loc: storage)
   local_opt, _, _ = opt_initialize(local_checkpoint, 'opennmt_translate_opt.pt', 'opennmt_translate_dummy_opt.pt')
   local_opt.gpuid = [gpu_id]
   _train = torch.load(TRAIN_DATA + '.train.pt')
   _valid = torch.load(TRAIN_DATA + '.valid.pt')
   local_fields = load_fields(_train, _valid, local_checkpoint, local_opt)
   #local_ref_model = init_train_model(local_checkpoint, local_opt, local_fields) # fields need data

   thenet = init_train_model(local_checkpoint, local_opt, local_fields) # fields need data
   pruned_model = apply_prune(thenet, local_ref_model_dict, local_sorted_weights, param_name, prune_threshold[rank])
   fitness = evaluate(pruned_model, _valid, local_fields, local_opt)
   fitness[2] = rank
   tensor_list = []

   if rank == 0: # master node
     tensor_list = [torch.FloatTensor([0.0,0.1,0.2]) for i in range(size)]
     dist.gather(fitness, gather_list = tensor_list)
     for ind_i in range(size):
        results[ind_i].copy_(tensor_list[ind_i])
   else:
     dist.gather(fitness, dst=0)
   #print(fitness)
   
def init_processes_trans(rank, size, param_name, references, prune_threshold, fn, model_dict, sorted_weights, trans_opt, trans_opt_dummy, results, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.2'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, param_name, references, prune_threshold, model_dict, sorted_weights, trans_opt, trans_opt_dummy, results)

def prune_and_eval_trans(rank, size, param_name, references, prune_threshold, ref_model_dict, ref_sorted_weights, trans_opt, trans_opt_dummy, results):
   local_ref_model_dict = ref_model_dict
   local_sorted_weights = ref_sorted_weights
   local_opt = trans_opt
   local_opt_dummy = trans_opt_dummy
   gpu_id = GPU_ID

   if rank>=5 and rank <size: # split tasks to different GPUs
     gpu_id = GPU_ID2
     cuda.set_device(gpu_id)

   local_opt.gpu = gpu_id

   thetranslator = init_translate_model( local_opt, local_opt_dummy) 
   
   apply_prune(thetranslator.model, local_ref_model_dict, local_sorted_weights, param_name, prune_threshold[rank])

   translate_data = onmt.IO.ONMTDataset(
        local_opt.src, local_opt.tgt, thetranslator.fields,
        use_filter_pred=False)

   prune_data = onmt.IO.OrderedIterator(
        dataset=translate_data, device=local_opt.gpu,
        batch_size=1, train=False, sort=False,
        shuffle=False)

   fitness = evaluate_trans(thetranslator, references, prune_data, translate_data)
   fitness[2] = rank
   tensor_list = []

   if rank == 0: # master node
     tensor_list = [torch.FloatTensor([0.0,0.1,0.2]) for i in range(size)]
     dist.gather(fitness, gather_list = tensor_list)
     for ind_i in range(size):
        results[ind_i].copy_(tensor_list[ind_i])
   else:
     dist.gather(fitness, dst=0)

# NCS
def NCS_MP(crates, ncs_stepsize, checkpoint, fields, opt, ref_net, ref_model_dicts, sorted_weights, param_name, train, valid, acc_constraint):
   popsize = 4
   __C = edict()
   __C.parameters = {'reset_xl_to_pop':False,'init_value':crates, 'stepsize':ncs_stepsize, 'bounds':crates, 'ftarget':0, 'tmax':100, 'popsize':popsize, 'best_k':1}
   es = ncs.NCS(__C.parameters)
   print('***************NCS initialization***************')
   tmp_fit = evaluate(ref_net, valid, fields, opt)
   es.set_initFitness(es.popsize*[sum(crates)+len(crates)]) # assume the inital crates store the size of each tensor
   print('fit:{}'.format(tmp_fit))
   print('***************NCS initialization***************')
   while not es.stop():
      X = es.ask()
      
      processes = []
      results = {}
      for ind_i in range(popsize):
        results[ind_i] = torch.FloatTensor([0.0,0.,0.])

      for rank in range(popsize):
        tmp_ref_model_dict = ref_model_dicts[0]
        tmp_sorted_weights = sorted_weights[0]
        if rank>=2 and rank <4: # split tasks to different GPUs
          tmp_ref_model_dict = ref_model_dicts[1]
          tmp_sorted_weights = sorted_weights[1]
        p = Process(target=init_processes, args=(rank, popsize, param_name, X, prune_and_eval, tmp_ref_model_dict, tmp_sorted_weights, results))
        p.start()
        processes.append(p)
      for p in processes:
         p.join()

      fit = []
      for i in range(len(X)):
        remain_num = sum(X[i])
        for j in range(len(results)): # results of fitness evaluation
           if int(results[j][2]) == i: # 0:ppl, 1:acc, 2:rank of individual
              if tmp_fit[1] - results[j][1] > acc_constraint:
                remain_num = np.inf
        fit.append(remain_num)
      #print X,fit
      es.tell(X, fit)
      es.disp(100)

   pruned_model = apply_prune(ref_net, ref_model_dicts[1], sorted_weights[1], param_name, es.result()[0][0])
   best_prune = evaluate(pruned_model, valid, fields, opt)
   print('Accuracy:{}=>{}, ppl:{}=>{}'.format(tmp_fit[1], best_prune[1], tmp_fit[0], best_prune[0]))
   return es.result()

def NCS_MP_trans(crates, ncs_stepsize, references, vali_data, vali_raw_data, ref_net, ref_model_dicts, sorted_weights, param_name, trans_opt, trans_opt_dummy, acc_constraint, num_runs=0):
   total_time = 0
   itr_count = 0
   popsize = 10
   __C = edict()
   __C.parameters = {'reset_xl_to_pop':False,'init_value':crates, 'stepsize':ncs_stepsize, 'bounds':crates, 'ftarget':0, 'tmax':400, 'popsize':popsize, 'best_k':1}
   es = ncs.NCS(__C.parameters)

   start_t = time.time()

   print('***************NCS initialization***************')
   tmp_fit = evaluate_trans(ref_net, references, vali_data, vali_raw_data)
   es.set_initFitness(es.popsize*[sum(crates)+len(crates)]) # assume the inital crates store the size of each tensor
   #tmp_fit = torch.FloatTensor([0,0,0])

   end_t = time.time()
   total_time = (end_t - start_t)

   print('fit:{}'.format(tmp_fit))
   print('time {}min elapse'.format(total_time/60.))
   print('***************NCS initialization***************')

   while not es.stop():
      start_t = time.time()

      X = es.ask()
      
      processes = []
      results = {}
      for ind_i in range(popsize):
        results[ind_i] = torch.FloatTensor([0.0,0.,0.])

      for rank in range(popsize):
        tmp_ref_model_dict = ref_model_dicts[0]
        tmp_sorted_weights = sorted_weights[0]
        if rank>=5 and rank <popsize: # split tasks to different GPUs
          tmp_ref_model_dict = ref_model_dicts[1]
          tmp_sorted_weights = sorted_weights[1]
        p = Process(target=init_processes_trans, args=(rank, popsize, param_name, references, X, prune_and_eval_trans, tmp_ref_model_dict, tmp_sorted_weights, trans_opt, trans_opt_dummy, results))
        p.start()
        processes.append(p)
      for p in processes:
         p.join()

      fit = []
      for i in range(len(X)):
        remain_num = sum(X[i])
        for j in range(len(results)): # results of fitness evaluation
           if int(results[j][2]) == i: # 0:ppl, 1:acc, 2:rank of individual
              if tmp_fit[1] - results[j][1] > acc_constraint:
                remain_num = np.inf
        fit.append(remain_num)
      #print X,fit
      es.tell(X, fit)
      es.disp(100)

      end_t = time.time()
      itr_count += 1
      itr_time = end_t -start_t
      total_time += itr_time
      print('total time {}min elapse, itr#{} cost {} min'.format(total_time/60., itr_count, itr_time/60.))

   pruned_model = apply_prune(ref_net, ref_model_dicts[1], sorted_weights[1], param_name, es.result()[0][0])
   best_prune = evaluate(pruned_model, valid, fields, opt)
   print('Accuracy:{}=>{}, ppl:{}=>{}'.format(tmp_fit[1], best_prune[1], tmp_fit[0], best_prune[0]))
   saved_model_name = 'the_pruned_deen_model_%s.pt' % num_runs
   torch.save(pruned_model, saved_model_name)
   return es.result(), saved_model_name

#------mian--------------
def main():


  # train model
  cuda.set_device(GPU_ID)
  '''
  checkpoint1 = torch.load(weights, map_location=lambda storage, loc: storage)
  train_opt, _, _ = opt_initialize(checkpoint1, 'opennmt_translate_opt.pt', 'opennmt_translate_dummy_opt.pt')
  # train data loading
  train = torch.load(train_opt.data + '.train.pt')
  valid = torch.load(train_opt.data + '.valid.pt')

  train_fields = load_fields(train, valid, checkpoint1, train_opt)
  ref_model1 = init_train_model(checkpoint1, train_opt, train_fields) # fields need data
  '''
  translate_opt, translate_dummy_opt = translate_opt_initialize('opennmt_translate_opt.pt', 'opennmt_translate_dummy_opt.pt')
  translator1 = init_translate_model(translate_opt, translate_dummy_opt) 
  ref_model1 = translator1.model

  ref_model_dict1 = {}
  for m_name, m_tensor in ref_model1.named_parameters():
     ref_model_dict1[m_name] = m_tensor
  param_name, init_threshold, sorted_weights1 = sorted_w(ref_model1)

  cuda.set_device(GPU_ID2)
  '''
  checkpoint2 = torch.load(weights, map_location=lambda storage, loc: storage)
  ref_model2 = init_train_model(checkpoint2, train_opt, train_fields) # fields need data
  '''
  translator2 = init_translate_model(translate_opt, translate_dummy_opt) 
  ref_model2 = translator2.model

  ref_model_dict2 = {}
  for m_name, m_tensor in ref_model2.named_parameters():
     ref_model_dict2[m_name] = m_tensor
  param_name, init_threshold, sorted_weights2 = sorted_w(ref_model2)

  cuda.set_device(GPU_ID)
  '''

  # test model
  checkpoint = torch.load(weights, map_location=lambda storage, loc: storage)
  _, translate_opt, translate_dummy_opt = opt_initialize(checkpoint, 'opennmt_translate_opt.pt', 'opennmt_translate_dummy_opt.pt')
  translate_models = [init_translate_model(translate_opt, translate_dummy_opt)]
  '''

  # testing data loading
  translate_data = onmt.IO.ONMTDataset(
        translate_opt.src, translate_opt.tgt, translator1.fields,
        use_filter_pred=False)

  prune_data = onmt.IO.OrderedIterator(
        dataset=translate_data, device=translate_opt.gpu,
        #batch_size=translate_opt.batch_size, train=False, sort=False,
        batch_size=1, train=False, sort=False,
        shuffle=False)

  # do training
  start_t = time.time()


  #training_model = apply_prune(training_model, ref_model_dict, sorted_weights, param_name, prune_threshold)
  #print 'Sparsity: {}'.format(get_sparsity(training_model))

  #evaluate(training_model, valid, train_fields, train_opt)
  '''
  best_found = NCS_MP(init_threshold, 8000000, checkpoint1, train_fields, train_opt, ref_model1, [ref_model_dict1,ref_model_dict2], [sorted_weights1,sorted_weights2], param_name, train, valid, 5)

  end_t = time.time()
  print('Pruning: {} s'.format((end_t - start_t)))
  print('Total parameters: {}'.format(sum(init_threshold)))
  print(best_found)
  '''
  #optim = build_optim(training_model, checkpoint, train_opt)
  #train_model(training_model, train, valid, train_fields, optim, train_opt)


  tt=open(translate_opt.tgt, 'r')
  references = [[t] for t in tt]

  best_found, saved_model = NCS_MP_trans(init_threshold, 8000000, references, prune_data, translate_data, translator1, [ref_model_dict1,ref_model_dict2], [sorted_weights1,sorted_weights2], param_name, translate_opt, translate_dummy_opt, 0.01)
  '''
  # evaluate one sentence
  hypothesis = []
  score_total = 0.
  num_word_total = 0
  for batch in prune_data:
     pred_batch, gold_batch, pred_scores, gold_scores, attn, src = translate_models[0].translate(batch, translate_data)
     score_total += sum([score[0] for score in pred_scores])
     num_word_total += sum(len(x) for x in batch.tgt[1:]) 
     hypothesis.extend([' '.join(x[0]) for x in pred_batch])
  print('BLEU[final, n-gram1,n-gram2,...]: {}'.format(bleu.corpus_bleu(hypothesis, references)))
  # training/validation 阶段的ppl计算在onmt/Trainer.py的Statisci()中；translating的ppl计算在 translate.py中的reprot_score函数里
  print('PPL: {}'.format(math.exp(-score_total/num_word_total)))
  '''
  end_t = time.time()
  print('Time: {} min'.format((end_t - start_t)/60.))

if __name__ == "__main__":
  main()
