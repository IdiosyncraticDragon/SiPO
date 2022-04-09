import copy
import math
from os import stat
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import solve_discrete_lyapunov
from torch import cuda

from package.data import Corpus

# data & model path
proj_root = '../'
data_path = '{}/data/penn'.format(proj_root)
model_path = '{}/model/original_model/language_model/{}'.format(
    proj_root, 'lm_model_orignal.pt')

GPU_ID = 2
if GPU_ID:
    cuda.set_device(GPU_ID)
SEQ_LEN = 35
TRAIN_BATCH_SIZE = 20
TEST_BATCH_SIZE = 10
GRAD_CLIP = 0.25
TRAIN_LOG_INTERVAL = 200

EVAL_FUNC = torch.nn.CrossEntropyLoss()

WEIGHT_MATRICE = ['rnn.weight_ih_l0', 'rnn.weight_hh_l0',
                  'rnn.weight_ih_l1', 'rnn.weight_hh_l1']
OTHER_MATRICE = ['encoder.weight',
                 'rnn.bias_ih_l0', 'rnn.bias_hh_l0',
                 'rnn.bias_ih_l1', 'rnn.bias_hh_l1',
                 'decoder.weight', 'decoder.bias']


def get_batch(source, i, evaluation=False):
    seq_len = min(SEQ_LEN, len(source) - 1 - i)
    data = torch.autograd.Variable(source[i:i+seq_len], volatile=evaluation)
    target = torch.autograd.Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.autograd.Variable:
        return torch.autograd.Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate_lm(model, data_source, corpus, eval_batch_size):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    total_acc = 0
    ppl = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, SEQ_LEN):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        _, preds = torch.max(output_flat, 1)
        total_acc += (preds == targets).data.cpu().sum()
        total_loss += len(data) * EVAL_FUNC(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    avg_loss = total_loss[0] / len(data_source)
    accuracy = total_acc / data_source.nelement()
    ppl = math.exp(avg_loss)
    # return total_loss[0] / len(data_source)
    return torch.FloatTensor([ppl, accuracy, avg_loss])


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    data = data.cuda(device=GPU_ID)
    return data


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


class LDSMaskedLM:

    def __init__(self, pretrained_model):
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

        self.row_LDS = []
        self.col_LDS = []

        self.row_dis_temp = np.zeros((200, 200))
        self.col_dis_temp = np.zeros((200, 200))

        self.sorted_params = None

    def cal_sorted_params(self):
        temp_dict = []
        for name, params in self.pretrained_model.named_parameters():
            if name in OTHER_MATRICE:
                temp_dict.append(params.view(-1))
        self.sorted_params, _ = torch.abs(
            torch.cat(temp_dict)).sort(descending=True)

    def cal_LDS(self):
        # Process the weight matrix into the corresponding shape in the paper
        # Notice:
        #   There are two LSTM weight matrices in pytorch, each weight matrix contains 4 small weight matrices, the shape is, (4*hidden_size, input_size) (4*hidden_size, hidden size)
        #   There are eight LSTM matrices in the paper, and the shape after stacking is, (input_size, hidden_size, 4), (hidden_size, input_size, 4)
        # What needs to be done:
        #   Reshape (view) the four concatenated matrices
        #   Stack the matrices together (the original paper is 8 weight matrices stacked together, here two layers of LSTM are considered together and 16 layers are stacked together)
        #   Since the original paper is different from each axis here, a transpose operation is performed (replaced with the same coordinate order as the paper)
        weight_cat = []
        for name, params in self.pretrained_model.named_parameters():
            if name in WEIGHT_MATRICE:
                weight_cat.append(params.view(
                    4, int(params.size()[0]/4), int(params.size()[1])))
        weight_cat = torch.cat(weight_cat, dim=0).data
        weight_cat = torch.transpose(weight_cat, 0, 2)

        # Calculate the lds of each row, and weight_cat is the weight cube stacked in the order of the dimensions of the paper
        weight_cat_row = weight_cat
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
            self.row_LDS.append([A, C])
        print('row lds finished. time {} seconds.'.format(time.time()-start_time))

        # The calculation process is the same as that of the row, but the dimensions are different, so the two dimensions of weight_cat are exchanged here, and the same calculation is performed
        weight_cat_col = torch.transpose(weight_cat, 0, 1)
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

            self.col_LDS.append([A, C])
        print('col lds finished. time {} seconds.'.format(time.time()-start_time))

    def cal_dis_temp(self):
        start_time = time.time()
        for i, lds1 in enumerate(self.row_LDS):
            for j, lds2 in enumerate(self.row_LDS):
                if j > i:
                    self.row_dis_temp[i, j] = matrin_dist_for_LDS(lds1, lds2)
                    self.row_dis_temp[j, i] = self.row_dis_temp[i, j]
            if i % 100 == 99:
                print('row {} fin. time {} seconds.'.format(
                    i, time.time()-start_time))

        start_time = time.time()
        for i, lds1 in enumerate(self.col_LDS):
            for j, lds2 in enumerate(self.col_LDS):
                if j > i:
                    self.col_dis_temp[i, j] = matrin_dist_for_LDS(lds1, lds2)
                    self.col_dis_temp[j, i] = self.col_dis_temp[i, j]
            if i % 100 == 99:
                print('col {} fin. time {} seconds.'.format(
                    i, time.time()-start_time))

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
        row_n_cluster = int(200 * row_ratio)
        col_n_cluster = int(200 * col_ratio)

        # Perform K_Medoids clustering on rows and columns separately to get the center point of each class
        start_time = time.time()
        row_left, _ = self.K_Medoids(self.row_LDS, row_n_cluster,
                                self.row_dis_temp, max_iter=300)
        print('row k medoids finished. time {} seconds.'.format(
            time.time()-start_time))
        start_time = time.time()
        col_left, _ = self.K_Medoids(self.col_LDS, col_n_cluster,
                                self.col_dis_temp, max_iter=300)
        print('col k medoids finished. time {} seconds.'.format(
            time.time()-start_time))

        # Update the mask matrix, keep only the center point, and assign 0 to all rows and columns of non-center points
        # Among them, since the rows and columns calculated in the paper are opposite to those here, the rows and columns here are also opposite
        # At the same time, since the weight matrix in LSTM is (4*hidden_size, input_size) (4*hidden_size, hidden size), for the first dimension, the corresponding four rows need to be cropped
        for name, params in self.pretrained_model.named_parameters():
            if name in WEIGHT_MATRICE:
                mask_temp = torch.cuda.ByteTensor(params.size()).fill_(1)
                for i in range(int(params.size()[0]/4)):
                    if i not in col_left:
                        mask_temp[i, :] = 0
                        mask_temp[i+int(params.size()[0]/4), :] = 0
                        mask_temp[i+2*int(params.size()[0]/4), :] = 0
                        mask_temp[i+3*int(params.size()[0]/4), :] = 0
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


def train_ts(tea_model, stu_model, ntokens, train_data, batch_size, bptt, corpus, grad_clip, log_interval, epoch):

    LR = 0.2
    if epoch >= 6:
        LR = 0.1

    fmse = 0.5
    fl2 = 10e-5
    softmax_loss = SoftmaxLoss()
    cmse_loss = CMSELoss(200, 10000)

    # Turn on training mode which enables dropout.
    stu_model.train()

    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden_tea = tea_model.init_hidden(batch_size)
    hidden_stu = stu_model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data_tea, targets_tea = get_batch(train_data, i, evaluation=True)
        data_stu, targets_stu = get_batch(train_data, i, evaluation=False)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden_tea = repackage_hidden(hidden_tea)
        hidden_stu = repackage_hidden(hidden_stu)

        output_tea, hidden_tea = tea_model(data_tea, hidden_tea)

        stu_model.zero_grad()
        output_stu, hidden_stu = stu_model(data_stu, hidden_stu)

        ls = softmax_loss(output_stu.view(-1, ntokens), targets_stu)
        lmse = cmse_loss(output_stu.view(-1, ntokens),
                         output_tea.view(-1, ntokens))
        ll2 = 0
        for name, params in stu_model.named_parameters():
            if name == 'decoder.weight':
                ll2 = fl2 * torch.sum(params ** 2)

        loss = ls + fmse*lmse + ll2
        loss.backward()

        torch.nn.utils.clip_grad_norm(stu_model.parameters(), grad_clip)

        for p in stu_model.parameters():
            p.data.add_(-LR, p.grad.data)

        total_loss += loss.data

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // bptt, LR,
                      elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    stu_model.eval()


def stats(t_model, data, corpus):
    start_time = time.time()
    get_sparity(t_model)
    tmp_fit = evaluate_lm(t_model, data, corpus, TEST_BATCH_SIZE)
    print(tmp_fit)
    print('time {} seconds.'.format(time.time()-start_time))


def main():
    # train/valid/test data，加载数据集
    print('loading data...')
    start_time = time.time()
    corpus = Corpus(data_path)
    train_data = batchify(corpus.train, TRAIN_BATCH_SIZE)
    valid_data = batchify(corpus.valid, TEST_BATCH_SIZE)
    test_data = batchify(corpus.test, TEST_BATCH_SIZE)
    ntokens = len(corpus.dictionary)
    print('data loaded. time {} seconds.'.format(time.time()-start_time))

    # Load the pre-trained model, where the code does not involve retraining, so the model mode can remain eval
    print('load pretrained model...')
    start_time = time.time()
    pretrained_model = torch.load(
        model_path, map_location=lambda storage, loc: storage.cuda(GPU_ID))
    pretrained_model.eval()

    # Use LDSMaskedLM to build the mask of the model and get the masked model at the same time
    # The initial mask is all 1, that is to say, after executing mask_pruned_model, the pruned model here is consistent with the original model (that is, it is not pruned)
    lds_model = LDSMaskedLM(pretrained_model)
    lds_model.mask_pruned_model()
    print('model loaded. time {} seconds.'.format(time.time()-start_time))

    # before pruned，Evaluate metrics for models that have not been pruned
    print('eval before prunning...')
    stats(lds_model.pruned_model, valid_data, corpus)

    lds_model.cal_LDS()
    lds_model.cal_dis_temp()
    lds_model.cal_sorted_params()

    base_model = copy.deepcopy(lds_model.pruned_model)
    stat(base_model, valid_data, corpus)

    lds_model.prun_rnn(0.9, 0.9)
    lds_model.prun_other(0.9)
    lds_model.mask_pruned_model()
    stats(lds_model.pruned_model, valid_data, corpus)

    # retrain
    for i in range(8):
        train_ts(base_model, lds_model.pruned_model, ntokens, train_data, TRAIN_BATCH_SIZE, SEQ_LEN, corpus, GRAD_CLIP, TRAIN_LOG_INTERVAL, i)
        stats(lds_model.pruned_model)
        lds_model.mask_pruned_model()
        stats(lds_model.pruned_model)

    model_1 = copy.deepcopy(lds_model.pruned_model)

    model_1.save_pruned_model('model_temp.pt')


if __name__ == "__main__":
    main()
