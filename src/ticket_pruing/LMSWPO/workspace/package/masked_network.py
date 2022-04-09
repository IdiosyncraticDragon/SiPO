import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import cuda
import torch.distributed as dist
#import copy
from torch.nn.parallel import replicate
import pdb

def my_replicate(model, source_device_id, target_device_id):
    """
      1. deep copy the mode to gpu@device_id
      2. eihter .cuda() and replicate objects among gpus,
         but Module.cuda() will edit the origianl object
         function relicate can copy multiply copies, so it returns a list; Here, we only generate one copy
      3. the original replicate function won't generate new copies if target_devcie_id == source_device_id. This assist function fix that.
    """
    #   [source_device_id, target1, ...]
    #   return the models identical to the devices list including input
    with cuda.device(target_device_id):
       copies = replicate(model, [source_device_id, target_device_id]) 
       orig_copy = copies[0]
       new_copy = copies[1]
       if source_device_id == target_device_id:
         for param_name, module_tensor in new_copy.named_parameters():
            module_tensor.data = module_tensor.data.new(module_tensor.data.size()).copy_(module_tensor.data)
    del orig_copy
    return new_copy

class MaskedModel(nn.Module):
   """
     add masks to all the parameters of an existing model
   """
   def __init__(self, pretrained, group_dict, source_device_id, target_device_id):
     super(MaskedModel, self).__init__()
     self.skip_mark = 'skip'#skip the layer while pruning
     self.mask_dict = {}
     #self.sort_tensors = {}
     self.layer_element_num = []
     self.layer_num = 0
     self.layer_name_dict = {}
     self.sparsity = 1.0
     self.total_parameter_num = 0

     # used for group parameters
     self.group_name_list = []
     self.map_dict = {}
     self.group_num_dict = {}
     self.group_parameter_dict = {}
     self.group_threshold_list = {}

     # gpu realted
     self.sgpu_id = source_device_id
     self.tgpu_id = target_device_id

     # init group dicts
     self.group_name_list = [k for k in group_dict.keys()]# the indices of list will map the the thresholds list which is accepted at pruning
     self.group_threshold_list = len(self.group_name_list)*[0.]
     for key, layer_names in group_dict.items():
        self.group_num_dict[key] = 0
        # maping the layer name to group name
        for layer_name in layer_names:
           self.map_dict[layer_name] = key

     pretrained_model_on_device = None
     # for each retrieval, transfer the pretrained model to a dictionary
     if source_device_id == target_device_id:
        pretrained_model_on_device  = pretrained
     else:
        pretrained_model_on_device = my_replicate(pretrained, source_device_id, target_device_id) #[source_device_id, target1, ...]

     with cuda.device(target_device_id):
        self.pretrained_model_dict = dict([(n,v) for n,v in pretrained_model_on_device.named_parameters()])
        for param_name, module_param in self.pretrained_model_dict.items():
           if param_name in self.map_dict: # ignore no-grouped layers
             self.group_num_dict[self.map_dict[param_name]] += module_param.nelement()
        for group_name in self.group_name_list:
           self.group_parameter_dict[group_name] = torch.cuda.FloatTensor(1, self.group_num_dict[group_name])
        self.generate_mask(self.pretrained_model_dict) # init self.mask_dict
        self.masked_model = my_replicate(pretrained, source_device_id, target_device_id)
        self.generator = self.masked_model.generator
        self.encoder = self.masked_model.encoder
        self.decoder = self.masked_model.decoder

   # just run once, to init the masks
   def generate_mask(self, pretrained):
      if len(self.mask_dict) > 0:
        return

      start_idx_dict = {}
      for group_name in self.group_name_list:
         start_idx_dict[group_name] = 0

      for param_name, module_tensor in pretrained.items():
         if param_name in self.map_dict: # ignore no-grouped layers
           tmp_group_name = self.map_dict[param_name]
           '''
           self.mask_dict[param_name] = Variable(torch.cuda.FloatTensor(module_tensor.size()).fill_(1))
           '''
           self.mask_dict[param_name] = Variable(torch.cuda.ByteTensor(module_tensor.size()).fill_(1))
           '''
           self.sort_tensors[param_name],_ = torch.abs(module_tensor).view(-1).sort(descending=True)
           '''
           # fill the data to its group
           tmp_start_idx = start_idx_dict[tmp_group_name]
           index_tensor = torch.LongTensor()
           torch.arange(tmp_start_idx, tmp_start_idx + module_tensor.nelement(), out=index_tensor)
           self.group_parameter_dict[tmp_group_name].put_(index_tensor.cuda(self.tgpu_id), module_tensor.data, accumulate = False)
           start_idx_dict[tmp_group_name] = start_idx_dict[tmp_group_name] + module_tensor.nelement()
           # accumulate layer info
           self.layer_name_dict[param_name] = self.layer_num
           self.layer_num += 1
         self.layer_element_num.append(module_tensor.nelement())

      # abs and sort the goup parameters
      for group_name in self.group_name_list:
         module_tensor = self.group_parameter_dict[group_name]
         self.group_parameter_dict[group_name],_ = torch.abs(module_tensor).view(-1).sort(descending=True)

      self.total_parameter_num = sum(self.layer_element_num)

   def apply_mask(self):
      assert self.mask_dict is not None
      assert self.pretrained_model_dict is not None
      assert self.masked_model is not None

      for param_name, module_tensor in self.masked_model.named_parameters():
         if param_name in self.map_dict: # ignore no-grouped layers
            # clear the selected masked layer
            module_tensor.data.zero_()
            # generate masked model by applying masks on the pretrained model
            '''
            module_tensor.data.addcmul_(1.0, self.pretrained_model_dict[param_name].data, self.mask_dict[param_name].data)
            '''
            tmp_mask = self.mask_dict[param_name].data
            tmp_remain_value = self.pretrained_model_dict[param_name].data.masked_select(tmp_mask)
            module_tensor.data.masked_scatter_(tmp_mask, tmp_remain_value)

   # require a prune function prune_fn:
   #   input: masks_dict, pretrained_dict, sort_tensors, layer_name_list
   #   output: None
   #   effect: editing the masks according to 
   #           the operation defined in prune_fn
   def change_mask(self, threshold, prune_fn):
      for i in range(len(self.group_name_list)):
         tmp_group_name = self.group_name_list[i]
         self.group_threshold_list[i] = threshold[i]
      # apply
      prune_fn(threshold, self.mask_dict, self.pretrained_model_dict, self.map_dict, self.group_parameter_dict, self.group_name_list)

   def clear_cache(self):
      with cuda.device(self.tgpu_id):
        cuda.empty_cache()

   def return_threshold_and_group_name(self):
      return self.group_threshold_list, self.group_name_list

   def forward(self, src, tgt, lengths, dec_state=None):
      return self.masked_model(src, tgt, lengths, dec_state)

   def number_of_layers(self):
      return int(self.layer_num)

   def total_parameters_of_pretrain(self):
      if self.total_parameter_num == 0:
          self.total_parameter_num = sum(self.layer_element_num)
      return self.total_parameter_num

   def get_sparsity(self):
      remain_num = 0
      for param_name, module_tensor in self.masked_model.named_parameters():
         remain_num += torch.nonzero(module_tensor.data).size(0)
      self.sparsity = (self.total_parameters_of_pretrain() - remain_num)*1./self.total_parameters_of_pretrain()
      #print('remain: %d' % remain_num)
      #print('total: %d' % self.total_parameters_of_pretrain())
      return self.sparsity

