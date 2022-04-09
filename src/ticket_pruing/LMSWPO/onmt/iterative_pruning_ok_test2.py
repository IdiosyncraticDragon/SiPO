#encoding=utf-8
'''
Test for new masked_model.py
apply make_trainable() and make_evaluable()
'''

import sys
HOME_PATH='/home/lab_tk/liguiying'
DATA_PATH='/raid/lab_tk/liguiying'
MODEL_PATH='/raid/lab_tk/liguiying'
#reload(sys)
#sys.setdefaultencoding('utf-8')
sys.path.insert(0, '{}/gitProject/OpenNMT-py'.format(HOME_PATH))
sys.path.insert(0, '{}/gitProject/OpenNMT-py/workspace/package'.format(HOME_PATH))

from logger import Logger
import pdb

import os
import torch
from torch.multiprocessing import Process
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

import torch.nn as nn
from torch.autograd import Variable
from torch import cuda
import torch.distributed as dist
#import ncs
from easydict import EasyDict as edict
from onmt.Utils import use_gpu
import onmt
import bleu
import math
import time
import ncs
import numpy as np
import os,os.path,datetime
from masked_networkT import MaskedModel
import copy
from layer_group import group_dict #group_dict1, group_dict2

logger_path = '{}/test/ietrative-logs'.format(HOME_PATH)
logger = Logger(logger_path)

name_mark = sys.argv[1]
#weights='/home/lgy/deepModels/torch_models/opennmt-py/zh_bahdanau_acc_20.51_ppl_338.36_e1.pt'
#TRAIN_DATA  = '/home/lgy/data/wmt/wmt17-en-zh/pywmt17'

# ToDo 
TRAIN_DATA  = '{}/data/wmt/wmt14/opennmtdata/iterative/afterdeal/len50_pywmt14'.format(DATA_PATH)
SAVE_MODEL_PATH = './iterative_retrain_{}'.format(name_mark)
SAVE_MODEL_FOLDER = '{}/deepModels/torch_models/opennmt-py/prune/'.format(MODEL_PATH)
SAVE_MODEL_TMP_FOLDER ='{}/deepModels/torch_models/opennmt-py/prune_tmp/'.format(MODEL_PATH)
l=os.listdir(SAVE_MODEL_FOLDER)
l.sort(key=lambda fn: os.path.getmtime(SAVE_MODEL_FOLDER+fn) if not os.path.isdir(SAVE_MODEL_FOLDER+fn) else 0)
#weights=SAVE_MODEL_FOLDER+l[-1]
weights=SAVE_MODEL_FOLDER+l[-2]


GPU_ID = 13
other_GPU_IDs = [GPU_ID] # devices used for individuals, one device id for one individual
REPORT_EVERY = 5000
EPOCHS = 1
MAX_GRAD_NORM = 1
LEARNING_RATE = 1.0
START_DECAY_AT = 8
TEST_BATCH_SIZE= 64

def time_now():
    now = int(round(time.time()*1000))
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now/1000))


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
    # print(model)

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

def make_valid_data_iter(valid_data, batch_size=TEST_BATCH_SIZE, gpu_id=None):
    """
    This returns user-defined validate data iterator for the trainer
    to iterate over during each validate epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    is ok too.
    """
    return onmt.IO.OrderedIterator(
                dataset=valid_data, batch_size= batch_size,
                device= gpu_id if gpu_id is not None else GPU_ID,
                train=False, sort=True)

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

def train_model(model, train_data, valid_data, fields, optim, opt, num_runs, stop_acc):
    '''
    input: stop_acc, means stop in stop_acc
    output:
        recorverd, if successfully recorverd accuracy, return True
    '''
    recovered = False
    train_iter = make_train_data_iter(train_data, opt)
    # 两者接口不同
    valid_iter = make_valid_data_iter(valid_data, TEST_BATCH_SIZE)

    train_loss = make_loss_compute(model, fields["tgt"].vocab,
                                   train_data)
    valid_loss = make_loss_compute(model, fields["tgt"].vocab,
                                   valid_data)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches

    trainer = onmt.Trainer(model, train_iter, valid_iter,
                           train_loss, valid_loss, optim,
                           trunc_size, shard_size)
    # pdb.set_trace()
    # print(opt.start_epoch, opt.epochs)
    # print(opt.batch_size)
    # valid_stats = trainer.validate()
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        # return True
        print(time_now(), "epoch:", epoch)

        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())
        logger.scalar_summary('train_%s_ppl' % num_runs, train_stats.ppl(), num_runs*(opt.epochs+1)+epoch)
        logger.scalar_summary('train_%s_acc' % num_runs, train_stats.accuracy(), num_runs*(opt.epochs+1)+epoch)

        # 2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        logger.scalar_summary('train_%s_val_ppl' % num_runs, valid_stats.ppl(), num_runs*(opt.epochs+1)+epoch)
        logger.scalar_summary('train_%s_val_acc' % num_runs, valid_stats.accuracy(), num_runs*(opt.epochs+1)+epoch)

        # 3. Log to remote server.
        # if opt.exp_host:
        #     train_stats.log("train", experiment, optim.lr)
        #     valid_stats.log("valid", experiment, optim.lr)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        if valid_stats.accuracy() >= stop_acc:
            print("epoch %s recovered accuracy %s and new valid accuracy %s" % (epoch, stop_acc, valid_stats.accuracy()))
            return True
        # 5. Drop a checkpoint if needed.
        # if epoch >= opt.start_checkpoint_at:
        #     trainer.drop_checkpoint(opt, epoch, fields, valid_stats)
        # for test
        # return True
    return recovered
#----------------end OpenNMT-py----------------------------------

def translate_opt_initialize(trans_p, trans_dum_p):
    translate_opt = torch.load(trans_p)
    translate_dummy_opt = torch.load(trans_dum_p)
    #   translate
    translate_opt.model = weights
    #   dataset for pruning
    #translate_opt.src = '/home/lgy/data/wmt/wmt17-en-zh/smalltrain-test-en.txt.tok'
    #translate_opt.tgt = '/home/lgy/data/wmt/wmt17-en-zh/smalltrain-test-zh.txt.tok'
    translate_opt.src = '{}/data/wmt/wmt14/opennmtdata/iterative/origin/en-test.txt'.format(DATA_PATH)
    translate_opt.tgt = '{}/data/wmt/wmt14/opennmtdata/iterative/origin/de-test.txt'.format(DATA_PATH)
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
    train_opt.start_epoch = 1
    train_opt.epochs = train_opt.epochs
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


# Set the crates of each layer, 
# the pruning will happen in the next forward action
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

def evaluate(thenet, valid_data, fields, batch_size = TEST_BATCH_SIZE, gpu_id=None):
    '''
    translate_opt, translate_dummy_opt = translate_opt_initialize('opennmt_translate_opt.pt', 'opennmt_translate_dummy_opt.pt')
    translator = init_translate_model(translate_opt, translate_dummy_opt)
    del translator.model
    translator.model = thenet
    tt=open(translate_opt.tgt, 'r')
    references = [[t] for t in tt]
    translate_data = onmt.IO.ONMTDataset(
            translate_opt.src, translate_opt.tgt, fields,
            use_filter_pred=False)
    prune_data = onmt.IO.OrderedIterator(
            dataset=translate_data, device=gpu_id,
            batch_size=1, train=False, sort=False,
            shuffle=False)
    tmp_fit = evaluate_trans(translator, references, prune_data, translate_data)
    return tmp_fit# the last two 0.0 reserved for rank number, and sparsity
    '''
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
    print('BLEU: {}'.format(bleu_score))
    # training/validation 阶段的ppl计算在onmt/Trainer.py的Statisci()中；translating的ppl计算在 translate.py中的reprot_score函数里
    print('PPL: {}'.format(ppl))

    return torch.FloatTensor([ppl, bleu_score, 0.0])# the last reserved for rank number
# multi process
#------------------------------for parallel--------------------
def init_processes(rank, size, orig_fit, acc_constraint, fn, valid, es, masked_models, num_runs, final_results, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, orig_fit, acc_constraint, valid, es, masked_models[rank], num_runs, final_results)

def prune_and_eval(rank, size, orig_fit, acc_constraint, valid, es, ref_model, num_runs, final_results):
    _valid = valid
    gpu_id = GPU_ID
    total_iterations = es.Tmax/es.popsize
    individual_iter_count = 0
    #ref_model = masked_models[rank]
    X = torch.Tensor(copy.deepcopy(es.pop))
    communicate_size = es.n + 4 # the size of tensors transfer accross computers
    communicate_tensor = torch.FloatTensor(communicate_size*[0.])
    fitness_list = []
    itr_best_remain = 0

    if rank == 0: # rank 0 is the main process to collect finesses
        X.share_memory_()
        #fitness_list = [torch.FloatTensor([0.0,0.1,0.2,0.3]).share_memory_() for i in range(size)]
        fitness_list = [torch.FloatTensor(communicate_size*[0.]).share_memory_() for i in range(size)]

    if rank>=1 and rank <size: # split tasks to different GPUs
        gpu_id = other_GPU_IDs[rank-1]

    with cuda.device(gpu_id):
        local_fields = onmt.IO.load_fields(torch.load(TRAIN_DATA + '.vocab.pt'))
        _valid.fields = local_fields # fields can not be packed, so reconstruct it in each threahds

        while (individual_iter_count < total_iterations):
            if rank == 0: # master node
                itr_X = torch.Tensor(es.ask())
                # broadcast the fathers
                X.copy_(itr_X)
                dist.broadcast(itr_X, 0)
            else:
                # recieve fathers from the source process
                dist.broadcast(X, 0)

            # apply MP on model
            x = X.numpy()[rank]
            ref_model.change_mask(x, apply_MP_on_mask)
            
            ref_model.apply_mask()

            # evaluate pruned network
            fitness = evaluate(ref_model, _valid, local_fields)
            communicate_tensor[0] = fitness[0]
            communicate_tensor[1] = fitness[1]
            communicate_tensor[2] = rank
            communicate_tensor[3] = ref_model.get_sparsity()
            for i in range(x.size):
                communicate_tensor[i+4] = X[rank,i]#x[i]

            # sync fitness
            if rank == 0: # collect fitness across processes
                dist.gather(communicate_tensor, gather_list = fitness_list)
            else:
                dist.gather(communicate_tensor, dst = 0)

            # judge new solutions
            if rank == 0: # negatively correlated search in master node
                fit = []
                X_ = []
                for i in range(es.popsize):
                    the_fitness = 100
                    for j in range(len(fitness_list)): # results of fitness evaluation
                        if int(fitness_list[j][2]) == i: # 0:ppl, 1:acc, 2:rank of individual
                            X_.append(fitness_list[j].numpy()[4:])
                            if orig_fit[1] - fitness_list[j][1] <= acc_constraint:
                                the_fitness = -fitness_list[j][3]
                            else:
                                the_fitness = (orig_fit[1] - fitness_list[j][1])/acc_constraint
                            continue
                    fit.append(the_fitness)

                es.tell(X_, fit)

                itr_best_remain = min(fit)

            final_results['result_NCS'].copy_(torch.Tensor(es.result()[0]))
            individual_iter_count += 1

            if rank==0: # record status
                logger.scalar_summary('ncs_%s_fitness' % num_runs, es.result()[1], num_runs*total_iterations + individual_iter_count)
                logger.scalar_summary('ncs_%s_best_itr_remain' % num_runs, itr_best_remain, num_runs*total_iterations + individual_iter_count)
                logger.histo_summary('ncs_%s_pop' % num_runs, es.result()[0], num_runs*total_iterations + individual_iter_count)
                logger.histo_summary('pop of 1', X_[0], num_runs*total_iterations + individual_iter_count)
                logger.scalar_summary('sp of 1', -fitness_list[0][3], num_runs*total_iterations + individual_iter_count)
                logger.scalar_summary('rank of 1', fitness_list[0][2], num_runs*total_iterations + individual_iter_count)
                logger.histo_summary('pop of 2', X_[1], num_runs*total_iterations + individual_iter_count)
                logger.scalar_summary('sp of 2', -fitness_list[1][3], num_runs*total_iterations + individual_iter_count)
                logger.scalar_summary('rank of 2', fitness_list[1][2], num_runs*total_iterations + individual_iter_count)
                #logger.histo_summary('pop of 3', X_[2], num_runs*total_iterations + individual_iter_count)
                #logger.scalar_summary('sp of 3', -fitness_list[2][3], num_runs*total_iterations + individual_iter_count)
                #logger.scalar_summary('rank of 3', fitness_list[2][2], num_runs*total_iterations + individual_iter_count)

    ref_model.clear_cache()

    #print(fitness)

# NCS
def NCS_MP(crates, ncs_stepsize,  fields, masked_models, valid, acc_constraint, num_runs=0, checkpoint=None):
    total_time = 0
    total_iteration = 100
    itr_count = 0
    popsize = len(other_GPU_IDs) + 1
    __C = edict()
    __C.parameters = {'reset_xl_to_pop':False,'init_value':crates, 'stepsize':ncs_stepsize, 'bounds':[0., 0.95], 'ftarget':0, 'tmax':total_iteration*popsize, 'popsize':popsize, 'best_k':1}
    es = ncs.NCS(__C.parameters)

    start_t = time.time()

    print('***************NCS initialization***************')
    ref_net = masked_models[0]
    # 0.0 represents no parameters have been pruned, so it's original fitness
    ref_net.change_mask(len(crates)*[0.0], apply_MP_on_mask)
    ref_net.apply_mask()
    orignal_fit = evaluate(ref_net, valid, fields)
    print('original fit: {}'.format(orignal_fit))

    ref_net = masked_models[0]
    ref_net.change_mask(crates, apply_MP_on_mask)
    ref_net.apply_mask()
    tmp_fit = evaluate(ref_net, valid, fields)
    print('Start sparsity: {}%'.format(ref_net.get_sparsity()*100))
    es.set_initFitness(es.popsize*[ref_net.get_sparsity()]) # assume the inital crates store the size of each tensor
    #es.ask()
    #tmp_fit = torch.FloatTensor([0,0,0])

    end_t = time.time()
    total_time = (end_t - start_t)

    print('fit:{}'.format(tmp_fit))
    print('time {}min elapse'.format(total_time/60.))
    print('***************NCS initialization***************')

    ref_net.clear_cache()
    valid.fields = [] # clear fields for send valid among thresholds
    processes = []
    results = {'result_NCS':torch.FloatTensor(crates)}
    results['result_NCS'].share_memory_()

    # paralell individuals
    for rank in range(popsize):
        p = Process(target=init_processes, args=(rank, popsize, orignal_fit, acc_constraint, prune_and_eval, valid, es, masked_models, num_runs, results))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    valid.fields = fields
    ref_net.change_mask(results['result_NCS'].numpy(), apply_MP_on_mask)
    ref_net.apply_mask()
    best_prune = evaluate(ref_net, valid, fields)
    print('Accuracy:{}=>{}, ppl:{}=>{}, sparsity: {}%'.format(orignal_fit[1], best_prune[1], orignal_fit[0], best_prune[0], ref_net.get_sparsity()*100.))

    logger.scalar_summary('ncs_start_acc', tmp_fit[1], num_runs)
    logger.scalar_summary('ncs_start_ppl', tmp_fit[0], num_runs)
    logger.scalar_summary('ncs_best_acc', best_prune[1], num_runs)
    logger.scalar_summary('ncs_best_ppl', best_prune[0], num_runs)
    if checkpoint is not None:
        real_model = (ref_net.masked_model.module
                      if isinstance(ref_net.masked_model, nn.DataParallel)
                      else ref_net.masked_model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)
        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint['model'] = model_state_dict
        checkpoint['generator'] = generator_state_dict
        saved_model_name = 'the_pruned_deen_model_%s.pt' % num_runs
        torch.save(checkpoint, saved_model_name)

    return results['result_NCS'].numpy(), saved_model_name, ref_net.masked_model

def time_now():
  now = int(round(time.time()*1000))
  return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))

def train_opt_initialize(c_point):
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
    return train_opt

def update_checkpoint(checkpoint, ref_net, run_times):
    real_model = (ref_net.masked_model.module
                    if isinstance(ref_net.masked_model, nn.DataParallel)
                    else ref_net.masked_model)
    real_generator = (real_model.generator.module
                        if isinstance(real_model.generator, nn.DataParallel)
                        else real_model.generator)
    model_state_dict = real_model.state_dict()
    model_state_dict = {k: v for k, v in model_state_dict.items()
                        if 'generator' not in k}
    generator_state_dict = real_generator.state_dict()
    checkpoint['model'] = model_state_dict
    checkpoint['generator'] = generator_state_dict
    saved_model_name = 'the_pruned_deen_model_%s_%s.pt' % (name_mark, run_times)
    torch.save(checkpoint, SAVE_MODEL_TMP_FOLDER+saved_model_name)
    return checkpoint, saved_model_name

#------mian--------------
def main():

    total_times = 100
    run_times = 0
    init_threshold = ...
    start_t = time.time()

    valid_data = torch.load(TRAIN_DATA + '.valid.pt')
    fields = onmt.IO.load_fields(torch.load(TRAIN_DATA + '.vocab.pt'))
    # fields = onmt.IO.load_fields_from_vocab(torch.load(TRAIN_DATA + '.vocab.pt'))
    valid_data.fields = fields # we need to clear this assignment relationg if we want to transfere valid among threads

    checkpoint = torch.load(weights, map_location=lambda storage, loc: storage)
    model_opt = checkpoint['opt']
    masked_models = []
    with cuda.device(GPU_ID):
        ref_model = onmt.ModelConstructor.make_base_model(model_opt, fields, True, checkpoint)
        ref_model.eval()
        ref_model.generator.eval()
        masked_model = MaskedModel(ref_model, group_dict, cuda.current_device(), cuda.current_device()) # ref_model is at current_device, no copy will happen
        masked_models.append(masked_model)

    if GPU_ID:
        cuda.set_device(GPU_ID)

    # 1 means 1% acc
    acc_percent_prune = 1
    # 只需要原始的accuracy
    acc_of_no_prune = 0
    get_acc_of_no_prune = False
    print(time_now(), "start while")
    while run_times < total_times:
        print("-----------------------------------------")
        print("start Iteration ", run_times)
        # init threshold
        best_threshold = 0
        itr_time = time.time()
        '''
            display all the names of parameters
        '''
        '''
            aa=ref_model.named_parameters
            aa_namelist = [ak[0] for ak in aa]
        '''
        '''
            test MP
        '''
        translate_opt, translate_dummy_opt = translate_opt_initialize('opennmt_translate_opt.pt', 'opennmt_translate_dummy_opt.pt')
        translator = init_translate_model(translate_opt, translate_dummy_opt)
        del translator.model
        translator.model = masked_model
        tt=open(translate_opt.tgt, 'r')
        references = [[t] for t in tt]

        xxx=np.arange(0.,1, 0.01)
        #for i in range(len(masked_model.group_name_list)):
        #   tmp_crate = len(masked_model.group_name_list)*[0.]
        print(time_now(), "start testing pruning")
        masked_model.make_evaluable()
        for i in range(len(xxx)):
            # best_threshold = 0.55
            # break
            translate_data = onmt.IO.ONMTDataset(
                translate_opt.src, translate_opt.tgt, fields,
                use_filter_pred=False)
            prune_data = onmt.IO.OrderedIterator(
                dataset=translate_data, device=GPU_ID,
                batch_size=1, train=False, sort=False,
                shuffle=False)
            tmp_crate = len(masked_model.group_name_list)*[xxx[i]]
            #tmp_crate[i] = 0.01
            masked_model.change_mask(tmp_crate, apply_MP_on_mask)
            masked_model.apply_mask()
            tmp_fit = evaluate(masked_model, valid_data, fields)
            #tmp_fit = evaluate_trans(translator, references, prune_data, translate_data)
            #logger.scalar_summary('test_bleu', tmp_fit[1]*100, int(xxx[i]*100))
            #logger.scalar_summary('acc', tmp_fit[1], int(xxx[i]*100))
            #logger.scalar_summary('ppl', tmp_fit[0], int(xxx[i]*100))
            #logger.scalar_summary('test_ppl', tmp_fit[0], int(xxx[i]*100))
            #print('group %s => acc (%.4f), ppl (%.4f)' % (masked_model.group_name_list[i], tmp_fit[1], tmp_fit[0]))
            #print('percentage %s => bleu (%.4f), ppl (%.4f)' % (xxx[i]*100, tmp_fit[1]*100, tmp_fit[0]))
            # print('percentage %s => acc (%.4f), ppl (%.4f)' % (xxx[i]*100, tmp_fit[1], tmp_fit[0]))
            if i == 0 and not get_acc_of_no_prune:
                acc_of_no_prune = tmp_fit[1]
                acc_of_no_prune = int(acc_of_no_prune*100)/100
                get_acc_of_no_prune = True
            elif acc_of_no_prune - tmp_fit[1] > acc_percent_prune:
                best_threshold = xxx[i] - 0.01
                break
        # -------------------------------------------------
        # Start writing
        # prune again
        print(time_now(), " start accuracy:", acc_of_no_prune)
        print("-------test------------:", get_acc_of_no_prune)
        print(time_now(), " apply pruning with threshold:", best_threshold)
        tmp_crate = len(masked_model.group_name_list)*[best_threshold]
        masked_model.change_mask(tmp_crate, apply_MP_on_mask)
        masked_model.apply_mask()

        # print information
        tmp_fit = evaluate(masked_model, valid_data, fields)
        print('percentage %s => acc (%.4f), ppl (%.4f)' % (best_threshold*100, tmp_fit[1], tmp_fit[0]))
        model_sparsity = masked_model.get_sparsity()
        print('Sparsity: {}'.format(model_sparsity))

        #--------------- start retraining --------------
        # first store model
        print(time_now(),"start saving model")
        _, saved_model = update_checkpoint(checkpoint, masked_model, run_times)
        print(time_now(), "finish saving model")
        print(time_now(), "start loading model")
        checkpoint = torch.load(SAVE_MODEL_TMP_FOLDER+saved_model, map_location=lambda storage, loc: storage)
        train_opt, _, _ = opt_initialize(checkpoint, 'opennmt_translate_opt.pt', 'opennmt_translate_dummy_opt.pt')

        # train data loading
        print(time_now(), "start loading data for retraining")
        train = torch.load(train_opt.data+ '.train.pt')
        valid = torch.load(train_opt.data + '.valid.pt')
        print(time_now(), "finish data loading")
        
        train_fields = load_fields(train, valid, checkpoint, train_opt)
        model_for_train = init_train_model(checkpoint, train_opt, train_fields)
        masked_model = MaskedModel(model_for_train, group_dict, cuda.current_device(), cuda.current_device())
        
        masked_model.make_trainable()

        print(time_now(), "building optm")
        optim = build_optim(model_for_train, checkpoint, train_opt)

        print(time_now(), "start restraining")
        recovered = train_model(model_for_train, train, valid, train_fields, optim, train_opt, run_times, acc_of_no_prune)
        print(time_now(), "finish retraining ")
        if not recovered:
            exit()
        else:
            print("------------Accuracy recorverd!--------------------")
            print("recovered accuracy:", acc_of_no_prune)
        run_times += 1
        
        masked_model.make_evaluable()
        tmp_fit = evaluate(masked_model, valid_data, fields)
        print("------------------for test-------------------")
        print('percentage %s => acc (%.4f), ppl (%.4f)' % (best_threshold*100, tmp_fit[1], tmp_fit[0]))
    
if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn')
    main()
