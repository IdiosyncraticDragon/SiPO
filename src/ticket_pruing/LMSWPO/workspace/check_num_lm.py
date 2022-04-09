import model
import numpy as np
import torch

ref_m = None
model_path = '/raid/lab_tk/liguiying/deepModels/torch_models/language-model/model.pt'
with open(model_path,'rb') as ff:
   ref_m = torch.load(ff)
aaa = [x.numel() for a,x in ref_m.named_parameters()]
print("{} paramters in totall".format(np.sum(aaa)))
