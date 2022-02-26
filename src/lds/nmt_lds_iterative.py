import copy
import math
import os
import sys
import time

sys.path.insert(0, '../')
sys.path.insert(0, './package')
import bleu
import numpy as np
import onmt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import solve_discrete_lyapunov
from torch import cuda


# MODEL_TYPE = 'rnnsearch'
MODEL_TYPE = 'luognet'
GPU_ID = 3

# data path
proj_root = '../'
data_path = '{}/data/wmt14'.format(proj_root)
model_path = ''
if MODEL_TYPE == 'rnnsearch':
    model_path = '{}/model/original_model/RNNSearch/{}'.format(
        proj_root, 'rnnsearch_original.pt')
elif MODEL_TYPE == 'luognet':
    model_path = '{}/model/original_model/LuongNet/{}'.format(
        proj_root, 'luognet_original.pt')

WEIGHT_MATRICE = []
OTHER_MATRICE = []
if MODEL_TYPE == 'rnnsearch':
    WEIGHT_MATRICE = [
        'encoder.rnn.weight_ih_l0',
        'encoder.rnn.weight_hh_l0',
        'encoder.rnn.weight_ih_l0_reverse',
        'encoder.rnn.weight_hh_l0_reverse',
        'encoder.rnn.weight_ih_l1',
        'encoder.rnn.weight_hh_l1',
        'encoder.rnn.weight_ih_l1_reverse',
        'encoder.rnn.weight_hh_l1_reverse',
        'decoder.rnn.layers.0.weight_ih',
        'decoder.rnn.layers.0.weight_hh',
        'decoder.rnn.layers.1.weight_ih',
        'decoder.rnn.layers.1.weight_hh'
    ]
    OTHER_MATRICE = [
        'encoder.embeddings.make_embedding.emb_luts.0.weight',
        'encoder.rnn.bias_ih_l0',
        'encoder.rnn.bias_hh_l0',
        'encoder.rnn.bias_ih_l0_reverse',
        'encoder.rnn.bias_hh_l0_reverse',
        'encoder.rnn.bias_ih_l1',
        'encoder.rnn.bias_hh_l1',
        'encoder.rnn.bias_ih_l1_reverse',
        'encoder.rnn.bias_hh_l1_reverse',
        'decoder.embeddings.make_embedding.emb_luts.0.weight',
        'decoder.rnn.layers.0.bias_ih',
        'decoder.rnn.layers.0.bias_hh',
        'decoder.rnn.layers.1.bias_ih',
        'decoder.rnn.layers.1.bias_hh',
        'decoder.attn.linear_context.weight',
        'decoder.attn.linear_query.weight',
        'decoder.attn.linear_query.bias',
        'decoder.attn.v.weight ',
        'decoder.attn.linear_out.weight',
        'decoder.attn.linear_out.bias',
        'generator.0.weight',
        'generator.0.bias'
    ]
elif MODEL_TYPE == 'luognet':
    WEIGHT_MATRICE = [
        'encoder.rnn.weight_ih_l0',
        'encoder.rnn.weight_hh_l0',
        'encoder.rnn.weight_ih_l1',
        'encoder.rnn.weight_hh_l1',
        'encoder.rnn.weight_ih_l2',
        'encoder.rnn.weight_hh_l2',
        'encoder.rnn.weight_ih_l3',
        'encoder.rnn.weight_hh_l3',
        'decoder.rnn.layers.0.weight_ih',
        'decoder.rnn.layers.0.weight_hh',
        'decoder.rnn.layers.1.weight_ih',
        'decoder.rnn.layers.1.weight_hh',
        'decoder.rnn.layers.2.weight_ih',
        'decoder.rnn.layers.2.weight_hh',
        'decoder.rnn.layers.3.weight_ih',
        'decoder.rnn.layers.3.weight_hh',
    ]
    OTHER_MATRICE = [
        'encoder.embeddings.make_embedding.emb_luts.0.weight',
        'encoder.rnn.bias_ih_l0',
        'encoder.rnn.bias_hh_l0',
        'encoder.rnn.bias_ih_l1',
        'encoder.rnn.bias_hh_l1',
        'encoder.rnn.bias_ih_l2',
        'encoder.rnn.bias_hh_l2',
        'encoder.rnn.bias_ih_l3',
        'encoder.rnn.bias_hh_l3',
        'decoder.embeddings.make_embedding.emb_luts.0.weight',
        'decoder.rnn.layers.0.bias_ih',
        'decoder.rnn.layers.0.bias_hh',
        'decoder.rnn.layers.1.bias_ih',
        'decoder.rnn.layers.1.bias_hh',
        'decoder.rnn.layers.2.bias_ih',
        'decoder.rnn.layers.2.bias_hh',
        'decoder.rnn.layers.3.bias_ih',
        'decoder.rnn.layers.3.bias_hh',
        'decoder.attn.linear_in.weight',
        'decoder.attn.linear_out.weight',
        'generator.0.weight',
        'generator.0.bias'
    ]


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


def make_valid_data_iter(valid_data, batch_size, gpu_id=None):
    """
    This returns user-defined validate data iterator for the trainer
    to iterate over during each validate epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    is ok too.
    """
    return onmt.IO.OrderedIterator(
        dataset=valid_data, batch_size=batch_size,
        device=gpu_id if gpu_id is not None else GPU_ID,
        train=False)


def make_loss_compute(
        model,
        tgt_vocab,
        dataset,
        gpu_id=None,
        copy_attn=False,
        copy_attn_force=False):
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

    if gpu_id is None:
        gpu_id = cuda.current_device()
    compute.cuda(gpu_id)

    return compute


def evaluate(thenet, valid_data, fields, batch_size=64, gpu_id=None):
    gpu_used = gpu_id if gpu_id is not None else torch.cuda.current_device()
    valid_iter = make_valid_data_iter(valid_data, batch_size, gpu_used)
    valid_loss = make_loss_compute(
        thenet, fields["tgt"].vocab, valid_data, gpu_used)

    stats = Statistics()

    for batch in valid_iter:
        _, src_lengths = batch.src
        src = onmt.IO.make_features(batch, 'src')
        tgt = onmt.IO.make_features(batch, 'tgt')
        # F-prop through the model.
        outputs, attns, _ = thenet(src, tgt, src_lengths)
        # Compute loss.
        batch_stats = valid_loss.monolithic_compute_loss(batch, outputs, attns)
        # Update statistics.
        stats.update(batch_stats)

    # the last two 0.0 reserved for rank number, and sparsity
    return torch.FloatTensor([stats.ppl(), stats.accuracy()])


def evaluate_trans(thenet, references, vali_data, vali_raw_data):
    hypothesis = []
    score_total = 0.
    num_word_total = 0
    for batch in vali_data:
        pred_batch, gold_batch, pred_scores, gold_scores, attn, src = thenet.translate(
            batch, vali_raw_data)
        score_total += sum([score[0] for score in pred_scores])
        num_word_total += sum(len(x) for x in batch.tgt[1:])
        hypothesis.extend([' '.join(x[0]) for x in pred_batch])
    ppl = math.exp(-score_total/num_word_total)
    bleu_score = bleu.corpus_bleu(hypothesis, references)[0][0]

    # the last reserved for rank number
    return torch.FloatTensor([ppl, bleu_score, 0.0])


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
              (epoch, batch, n_batches,
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
        experiment.add_scalar_value(prefix + "_tgtper", self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


# Calculate the matrin distance for the matrix A and matrix C of the LDS
def matrin_dist_for_LDS(lds1, lds2):

    # According to the formula (9) of the paper, the final matrix A is obtained, and the dimension is (2D, 2D)
    A = torch.zeros((lds1[0].shape[0]*2, lds1[0].shape[1]*2))
    A[:lds1[0].shape[0], :lds1[0].shape[0]] = lds1[0]
    A[lds1[0].shape[0]:, lds1[0].shape[0]:] = lds2[0]

    # According to the formula (10) of the paper, the final matrix C is obtained, and the dimension is (D, 2D)
    C = torch.zeros((lds1[0].shape[0], lds1[0].shape[1]*2))
    C[:, :lds1[0].shape[0]] = lds1[1]
    C[:, lds1[0].shape[0]:] = lds2[1]

    # According to the paper, find the solution to the Lyapunov equation (A_t P A - P = -Q, where Q is C_t C)
    # First find Q in the Lyapunov equation, and then use the scipy package to solve to get P
    Q = torch.mm(C.transpose(0, 1), C)
    P = solve_discrete_lyapunov(A, Q)

    # According to the formula (8) of the paper, the P of the corresponding position is obtained
    P11 = P[:lds1[0].shape[0], :lds1[0].shape[0]]
    P12 = P[:lds1[0].shape[0], lds1[0].shape[0]:]
    P21 = P[lds1[0].shape[0]:, :lds1[0].shape[0]]
    P22 = P[lds1[0].shape[0]:, lds1[0].shape[0]:]

    # According to the formula (11) of the paper, the corresponding eigenvalues are obtained
    eig_P = np.matmul(np.linalg.inv(P11), P12)
    eig_P = np.matmul(eig_P, np.linalg.inv(P22))
    eig_P = np.matmul(eig_P, P21)
    eig_P = np.linalg.eig(eig_P)[0]

    # According to the paper formula (12), the final distance is obtained
    return np.sqrt(-np.log(np.prod(eig_P ** 2)))


class LDSMaskedNMT:

    def __init__(self, pretrained_model, model_type='rnnsearch'):
        assert model_type == 'rnnsearch' or model_type == 'luognet'

        self.model_type = model_type
        self.stack_num = 3 if model_type == 'rnnsearch' else 4

        # The initial pretrained model, and a dictionary of weight matrices to be pruned
        self.pretrained_model = pretrained_model
        self.pretrained_model_dict = dict(
            [(name, params) for name, params in self.pretrained_model.named_parameters()])
        # The model after pruning is consistent with the original model before pruning
        self.pruned_model = copy.deepcopy(pretrained_model)
        # The mask matrix corresponding to the weight matrix to be pruned
        self.mask_dict = {}
        for name, params in self.pretrained_model.named_parameters():
            self.mask_dict[name] = torch.cuda.ByteTensor(
                params.size()).fill_(1)

        self.row_LDS_dict = {}
        self.col_LDS_dict = {}

        self.row_dist_temp_dict = {}
        self.col_dist_temp_dict = {}

        self.sorted_params = None

    def cal_LDS_for_all_matrices(self):
        for name in WEIGHT_MATRICE:
            self.cal_LDS_for_one_matrices(name)

    def cal_LDS_for_one_matrices(self, name):

        params = self.pretrained_model_dict[name]
        print(name, params.shape)

        weight_cat = params.view(self.stack_num,
                                 int(params.size()[0] / self.stack_num),
                                 int(params.size()[1])).data
        weight_cat = torch.transpose(weight_cat, 0, 2)

        # Calculate the lds of each row, and weight_cat is the weight cube stacked in the order of the dimensions of the paper
        weight_cat_row = weight_cat
        row_LDS = []
        # According to formula (5) in the paper, finding wrc_bar is the average (since it is the average of all row vectors, the dimension of the average is 1)
        row_wrc_bar = torch.mean(weight_cat_row, 1, keepdim=True)
        # According to formula (6) in the paper, simultaneously find the matrix Y of each row
        # The dimension of the Y matrix obtained here is opposite to that of the paper, so transpose is required (the row corresponds to the correct one, but the matrix dimension corresponding to each row is opposite)
        # When calculating svd at the same time, the GPU calculation will report an error, so we give the Y matrix to the CPU
        Y = weight_cat_row - row_wrc_bar
        Y = Y.transpose(1, 2).cpu()
        # Compute matrix A and matrix C separately for each row
        start_time = time.time()
        for k in range(weight_cat_row.size()[0]):
            # Perform singular value decomposition on a row of matrix Y
            # The matrix U of singular value decomposition is the matrix C corresponding to this row
            C, S_vec, Vt = torch.svd(Y[k], some=False)
            # Since the S returned by torch.svd is just a vector, it needs to be converted into a diagonal matrix of the corresponding dimension
            S = torch.zeros(Y[k].size()[0], Y[k].size()[1])
            S[:len(S_vec), :len(S_vec)] = torch.diag(S_vec)
            # Calculate the matrix Z in the paper
            Z = torch.mm(S, Vt)

            # According to formula (7) in the paper, calculate the matrix A in the paper, where np.linalg.pinv is the pseudo-inverse of the calculated matrix
            A = torch.mm(Z[:, 1:], torch.from_numpy(np.linalg.pinv(Z[:, :-1])))

            # Obtain matrix A and matrix C corresponding to LDS
            row_LDS.append([A, C])
        print('row lds finished. time {} seconds.'.format(
            time.time() - start_time))
        self.row_LDS_dict[name] = row_LDS

        # The calculation process is the same as that of the row, but the dimensions are different, so the two dimensions of weight_cat are exchanged here, and the same calculation is performed
        weight_cat_col = torch.transpose(weight_cat, 0, 1)
        col_LDS = []
        col_wrc_bar = torch.mean(weight_cat_col, 1, keepdim=True)
        Y = weight_cat_col - col_wrc_bar
        Y = Y.transpose(1, 2).cpu()
        start_time = time.time()
        for k in range(weight_cat_col.size()[0]):
            C, S_vec, Vt = torch.svd(Y[k], some=False)
            S = torch.zeros(Y[k].size()[0], Y[k].size()[1])
            S[:len(S_vec), :len(S_vec)] = torch.diag(S_vec)
            Z = torch.mm(S, Vt)

            A = torch.mm(Z[:, 1:], torch.from_numpy(np.linalg.pinv(Z[:, :-1])))

            col_LDS.append([A, C])
        print('col lds finished. time {} seconds.'.format(
            time.time() - start_time))
        self.col_LDS_dict[name] = col_LDS

    def cal_dis_temp(self):
        start_time = time.time()
        for name in WEIGHT_MATRICE:
            if os.path.exists('./dist_temp_dict/{}_{}_{}.npy'.format(self.model_type, 'row', name)) and os.path.exists('./dist_temp_dict/{}_{}_{}.npy'.format(self.model_type, 'col', name)):
                self.row_dist_temp_dict[name] = np.load(
                    './dist_temp_dict/{}_{}_{}.npy'.format(self.model_type, 'row', name))
                self.col_dist_temp_dict[name] = np.load(
                    './dist_temp_dict/{}_{}_{}.npy'.format(self.model_type, 'col', name))
                print('{} skip.'.format(name))
                continue
            self.row_dist_temp_dict[name] = np.zeros((self.pretrained_model_dict[name].shape[1],
                                                      self.pretrained_model_dict[name].shape[1]))
            self.col_dist_temp_dict[name] = np.zeros((int(self.pretrained_model_dict[name].shape[0]/self.stack_num),
                                                      int(self.pretrained_model_dict[name].shape[0]/self.stack_num)))

            for i, lds1 in enumerate(self.row_LDS_dict[name]):
                for j, lds2 in enumerate(self.row_LDS_dict[name]):
                    if j > i:
                        self.row_dist_temp_dict[name][j, i] = self.row_dist_temp_dict[name][i, j] = matrin_dist_for_LDS(
                            lds1, lds2)

            for i, lds1 in enumerate(self.col_LDS_dict[name]):
                for j, lds2 in enumerate(self.col_LDS_dict[name]):
                    if j > i:
                        self.col_dist_temp_dict[name][j, i] = self.col_dist_temp_dict[name][i, j] = matrin_dist_for_LDS(
                            lds1, lds2)

            np.save('./dist_temp_dict/{}_{}_{}.npy'.format(self.model_type,
                    'row', name), self.row_dist_temp_dict[name])
            np.save('./dist_temp_dict/{}_{}_{}.npy'.format(self.model_type,
                    'col', name), self.col_dist_temp_dict[name])
            print('{} {}'.format(name, time.time()-start_time))

    def cal_sorted_params(self):
        temp_dict = []
        for name, params in self.pretrained_model.named_parameters():
            if name in OTHER_MATRICE:
                temp_dict.append(params.view(-1))
        self.sorted_params, _ = torch.abs(
            torch.cat(temp_dict)).sort(descending=True)

    def K_Medoids(self, LDS, n_cluster, dis_temp, max_iter=np.inf):
        #     medoids_inds = np.random.choice(len(LDS), size=n_cluster, replace=False)
        # Originally, the center point should be taken randomly, but randomly taking the initial center point will lead to inconsistent results of each clustering, so it is simplified here, and the first few points are directly taken as the initial center point
        medoids_inds = np.arange(n_cluster)
        labels = np.zeros(len(LDS))

        converaged = False
        n_iter = 1
        while n_iter <= max_iter and not converaged:
            old_mediods = copy.deepcopy(medoids_inds)

            # compute all the dist from point to mediods
            dist_mat = np.zeros((len(LDS), n_cluster))
            for i, lds in enumerate(LDS):
                for j, ind in enumerate(old_mediods):
                    if i != ind:
                        dist_mat[i, j] = dis_temp[i, ind]

            # According to the calculated distance, assign center points to all points
            labels = np.argmin(dist_mat, axis=1)

            # update new medoids
            # According to the clustering results, for each class:
            #   Calculate the total distance of each point in each class to other points
            #   The point with the shortest total distance is selected as the new center
            for cluster_i in range(n_cluster):
                mat_inds = np.where(labels == cluster_i)[0]
                best_dist_sum = np.inf
                for mat_ind1 in mat_inds:
                    dist_sum = 0
                    for mat_ind2 in mat_inds:
                        if mat_ind1 != mat_ind2:
                            dist_sum += dis_temp[mat_ind1, mat_ind2]
                    if dist_sum < best_dist_sum:
                        best_dist_sum = dist_sum
                        medoids_inds[cluster_i] = mat_ind1

            # compare coverage
            converaged = set([int(ind) for ind in old_mediods]) == set(
                [int(ind) for ind in medoids_inds])
            n_iter += 1
#             print(n_iter, old_mediods, medoids_inds, converaged)
            print(n_iter, converaged)

        return medoids_inds, labels

    def prun_rnn(self, row_ratio, col_ratio):
        for name in WEIGHT_MATRICE:
            self.prun_one_rnn(name, row_ratio, col_ratio)

    def prun_one_rnn(self, name, row_ratio, col_ratio):
        params = self.pretrained_model_dict[name]
        row_n_cluster = int(params.size()[1] * row_ratio)
        col_n_cluster = int(params.size()[0] / self.stack_num * col_ratio)

        # Perform K_Medoids clustering on rows and columns separately to get the center point of each class
        start_time = time.time()
        row_left, _ = self.K_Medoids(self.row_LDS_dict[name], row_n_cluster,
                                     self.row_dist_temp_dict[name], max_iter=300)
        print('row k medoids finished. time {} seconds.'.format(
            time.time() - start_time))
        start_time = time.time()
        col_left, _ = self.K_Medoids(self.col_LDS_dict[name], col_n_cluster,
                                     self.col_dist_temp_dict[name], max_iter=300)
        print('col k medoids finished. time {} seconds.'.format(
            time.time() - start_time))

        # Update the mask matrix, keep only the center point, and assign 0 to all rows and columns of non-center points
        # Among them, since the rows and columns calculated in the paper are opposite to those here, the rows and columns here are also opposite
        mask_temp = torch.cuda.ByteTensor(params.size()).fill_(1)
        for i in range(int(params.size()[0] / self.stack_num)):
            if i not in col_left:
                for j in range(self.stack_num):
                    mask_temp[i + j *
                              int(params.size()[0]/self.stack_num), :] = 0
        for j in range(int(params.size()[1])):
            if j not in row_left:
                mask_temp[:, j] = 0
        self.mask_dict[name] = mask_temp

    def prun_other(self, ratio):
        threshold = self.sorted_params[int(len(self.sorted_params)*ratio)]

        for name, params in self.pretrained_model.named_parameters():
            if name in OTHER_MATRICE:
                self.mask_dict[name] = (
                    params.gt(threshold)+params.lt(threshold.neg())).data

    # Use the mask matrix to update the weight matrix of the pruned model
    def mask_pruned_model(self):
        for name, params in self.pruned_model.named_parameters():
            temp_remain_value = self.pretrained_model_dict[name].data.masked_select(
                self.mask_dict[name])
            params.data.zero_()
            params.data.masked_scatter_(
                self.mask_dict[name], temp_remain_value)

    # Save the model of the pruned matrix
    def save_pruned_model(self, path):
        torch.save(self.pruned_model, path)


def get_sparity(model):
    total_param = 0
    nonzero_param = 0

    for name, param in model.named_parameters():
        total_param += np.prod(param.shape)
        nonzero_param += torch.nonzero(param.data).shape[0]

    print('{} {} {}%'.format(total_param,
          nonzero_param, nonzero_param/total_param*100))


def test_metrics(t_model, fields):
    start_time = time.time()
    translate_opt, translate_dummy_opt = translate_opt_initialize(
        '../param/opennmt_translate_opt.pt', '../param/opennmt_translate_dummy_opt.pt', data_path, model_path, GPU_ID)
    translator = init_translate_model(translate_opt, translate_dummy_opt)
    del translator.model
    translator.model = t_model
    tt = open(translate_opt.tgt, 'r')
    references = [[t] for t in tt]
    print('translator loaded.')

    translate_data = onmt.IO.ONMTDataset(
        translate_opt.src, translate_opt.tgt, fields,
        use_filter_pred=False)
    prune_data = onmt.IO.OrderedIterator(
        dataset=translate_data, device=GPU_ID,
        batch_size=1, train=False, sort=False,
        shuffle=False)
    print('data loaded.')

    tmp_fit = evaluate_trans(translator, references,
                             prune_data, translate_data)
    print(tmp_fit)
    print('time {}.'.format(time.time()-start_time))


class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()

    def forward(self, input, target):
        softmax_input = F.softmax(input)
        return F.cross_entropy(softmax_input, target, None, True,
                               -100, True)


class CMSELoss(nn.Module):
    def __init__(self, cin, cout):
        super(CMSELoss, self).__init__()
        self.factor = 1. / (cin * cout)

    def forward(self, input, target):
        return self.factor * F.mse_loss(input, target, size_average=True, reduce=True)


def train_ts(tea_model, stu_model, train_data, epoch, batch_size=64, gpu_id=None):
    gpu_used = gpu_id if gpu_id is not None else torch.cuda.current_device()
    train_iter = make_valid_data_iter(train_data, batch_size, gpu_used)

    LR = 0.2
    if epoch >= 6:
        LR = 0.1

    fmse = 0.5
    fl2 = 10e-5
    softmax_loss = SoftmaxLoss()
    cmse_loss = CMSELoss(1000, 50004)

    # Turn on training mode which enables dropout.
    stu_model.train()
    stu_model.generator.train()
    tea_model.eval()
    tea_model.generator.train()

    total_loss = 0
    start_time = time.time()
    counter = 0
    for batch in train_iter:
        _, src_lengths = batch.src
        src = onmt.IO.make_features(batch, 'src')
        tgt = onmt.IO.make_features(batch, 'tgt')

        outputs, attns, _ = tea_model(src, tgt, src_lengths)
        outputs_tea = tea_model.generator(outputs).view(-1, 50004)

        stu_model.zero_grad()
        src.volatile = False
        tgt.volatile = False
        outputs, attns, _ = stu_model(src, tgt, src_lengths)
#         print(outputs.requires_grad, outputs.volatile)
        outputs_stu = stu_model.generator(outputs).view(-1, 50004)

        targets_stu = batch.tgt[1:].view(-1)

#         print(outputs_stu.requires_grad, outputs_stu.volatile)
#         print(targets_stu.requires_grad, targets_stu.volatile)
#         print(outputs_tea.requires_grad, outputs_tea.volatile)
        ls = softmax_loss(outputs_stu, targets_stu)
        lmse = cmse_loss(outputs_stu, outputs_tea)

        ll2 = 0
        for name, params in stu_model.named_parameters():
            #             print(name, params.requires_grad, params.volatile)
            if name == 'generator.0.weight':
                ll2 = fl2 * torch.sum(params ** 2)

        loss = ls + fmse*lmse + ll2
#         print(loss.requires_grad, loss.volatile)
        loss.backward()
#         break

        torch.nn.utils.clip_grad_norm(stu_model.parameters(), 0.25)

        for p in stu_model.parameters():
            p.data.add_(-LR, p.grad.data)

        total_loss += loss.data

        counter += 1
        if counter % 100 == 0:
            print('epoch {}, batch {}/{}, avg time {}, loss {}'.format(epoch, counter,
                                                                       len(train_data)//batch_size,
                                                                       (time.time(
                                                                       )-start_time)/100,
                                                                       total_loss[0]/100))
        total_loss = 0
        start_time = time.time()

    stu_model.eval()
    stu_model.generator.eval()


def main():
    # load valid data
    cuda.set_device(GPU_ID)
    print('loading data...')
    start_time = time.time()
    valid_data = torch.load(os.path.join(data_path, 'len50_pywmt14.valid.pt'))
    fields = onmt.IO.load_fields(torch.load(
        os.path.join(data_path, 'len50_pywmt14.vocab.pt')))
    valid_data.fields = fields
    print('data loaded. time {} seconds.'.format(time.time()-start_time))

    # load model
    print('load pretrained model...')
    start_time = time.time()
    checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    model_opt = checkpoint['opt']
    with cuda.device(GPU_ID):
        ref_model = onmt.ModelConstructor.make_base_model(
            model_opt, fields, True, checkpoint)
        ref_model.eval()
        ref_model.generator.eval()

    lds_model = LDSMaskedNMT(ref_model, MODEL_TYPE)
    lds_model.mask_pruned_model()
    print('model loaded. time {} seconds.'.format(time.time()-start_time))

    lds_model.cal_LDS_for_all_matrices()
    lds_model.cal_dis_temp()
    lds_model.cal_sorted_params()

    tmp_fit = evaluate(lds_model.pruned_model, valid_data, fields)
    print(tmp_fit)
    test_metrics(lds_model.pruned_model, fields)
    get_sparity(lds_model.pruned_model)

    tea_model = copy.deepcopy(lds_model.pruned_model)

    lds_model.prun_rnn(1.0, 1.0)
    lds_model.prun_other(0.6)

    lds_model.mask_pruned_model()
    tmp_fit = evaluate(lds_model.pruned_model, valid_data, fields)
    print(tmp_fit)
    test_metrics(lds_model.pruned_model, fields)

    stu_model = copy.deepcopy(lds_model.pruned_model)

    # load train data
    cuda.set_device(GPU_ID)
    print('loading data...')
    start_time = time.time()
    train_data = torch.load(os.path.join(data_path, 'len50_pywmt14.train.pt'))
    train_fields = onmt.IO.load_fields(torch.load(
        os.path.join(data_path, 'len50_pywmt14.vocab.pt')))
    train_data.fields = train_fields
    print('data loaded. time {} seconds.'.format(time.time()-start_time))

    # retrain
    train_ts(tea_model, stu_model, train_data, epoch=1)

    tmp_fit = evaluate(stu_model, valid_data, fields)
    print(tmp_fit)
    test_metrics(stu_model, fields)
    get_sparity(stu_model)

    lds_model.mask_pruned_model()
    tmp_fit = evaluate(stu_model, valid_data, fields)
    print(tmp_fit)
    test_metrics(stu_model, fields)
    get_sparity(stu_model)


if __name__ == "__main__":
    main()
