from easydict import EasyDict as edict
import os.path as osp

__C = edict() 

cfg = __C

__C.HOME_PATH = '/home/lgy'
__C.DATA_PATH = '/home2/lgy'
__C.MODEL_PATH = '/home2/lgy'
__C.PROJECT_ROOT = osp.join(osp.dirname(__file__),'..')
__C.PROJECT_LOG = osp.join(__C.PROJECT_ROOT,'logs')
__C.PROJECT_RECORD = osp.join(__C.PROJECT_ROOT,'running_records')
__C.PROJECT_PACKAGE = osp.join(osp.dirname(__file__),'package')

# for language model
__C.LM_LOG_MARK = 'lmrnn-logs'
#   for pruned model
__C.LM_MODEL_PATH = '{}/deepModels/torch_models/language-model/prune/'.format(__C.MODEL_PATH)
#   for retrained model
__C.LM_MODEL_TMP_FOLDER = '{}/deepModels/torch_models/language-model/prune_tmp/'.format(__C.MODEL_PATH)
