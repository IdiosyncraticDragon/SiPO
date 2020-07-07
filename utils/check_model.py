from model_inspect import *
from torch import cuda
import argparse

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="inspect the status of models") 
   parser.add_argument('--model', choices=['lm','rnnsearch', 'luongnet'], default='lm')
   parser.add_argument('--status', choices=['orig','pruned'], default='orig')
   #parser.add_argument('--opt', choices=['num_param','sparsity', 'valid_ppl', 'test_ppl', 'test_bleu', 'valid_acc', 'test_acc'], default='num_param')
   args = parser.parse_args()
   GPU_ID = 0

   # modify the path by the users
   lm_data_path = '../data/penn/'
   lm_model_path_orig = '../model/original_model/language_model/lm_model_orignal.pt'
   lm_model_path_pruned = '../model/pruned_model/language_model/lm_model_pruned.pt'

   wmt_data_path = '../data/wmt14/'
   rnnsearch_path_orig = '../model/original_model/RNNSearch/rnnsearch_original.pt'
   rnnsearch_path_pruned = '../model/pruned_model/RNNSearch/rnnsearch_pruned.pt'
   luongnet_path_orig = '../model/original_model/LuongNet/luognet_original.pt'
   luongnet_path_pruned = '../model/pruned_model/LuongNet/luongnet_pruned.pt'
   translate_param1_path = '../param/opennmt_translate_opt.pt'
   translate_param2_path = '../param/opennmt_translate_dummy_opt.pt'

   # procedure for language models
   if args.model == 'lm':
     print("========Language Model==========")
     corpus, _, val_data, test_data = lm_data_load(lm_data_path)
     if args.status == 'orig':
         model = lm_model_load(lm_model_path_orig)
         total_num = lm_original_param_num(model)
         print("The original model inspection")
         print("{} paramters in totall".format(total_num))
         val_ppl, val_acc = lm_evaluate(model, corpus, val_data)
         test_ppl, test_acc = lm_evaluate(model, corpus, test_data)
         print("Valid ppl: {}".format(val_ppl))
         print("Test ppl: {}".format(test_ppl))
         print("Valid acc: {}\%".format(100. * val_acc))
         print("Test acc: {}".format(100. * test_acc))
     elif args.status == 'pruned':
         model = lm_model_load(lm_model_path_pruned)
         total_num = lm_original_param_num(model.masked_model)
         sparsity = model.get_sparsity()
         print("The pruned model inspection")
         print("{} paramters in totall".format(total_num))
         print("{} paramters prund".format(int(sparsity * total_num)))
         print("sparsity: {}%".format(100. * sparsity))
         val_ppl, val_acc = lm_evaluate(model.masked_model, corpus, val_data)
         test_ppl, test_acc = lm_evaluate(model.masked_model, corpus, test_data)
         print("Valid ppl: {}".format(val_ppl))
         print("Test ppl: {}".format(test_ppl))
         print("Valid acc: {}%".format(100. * val_acc))
         print("Test acc: {}%".format(100. * test_acc))
     else:
         print("ERROR OPTION OF OPERATION")
   elif args.model == 'rnnsearch':
     print("========NMT Model: RNNSearch==========")
     from rnnsearch_layer_group import group_dict 
     if args.status == 'orig':
       total_param, sparsity, tmp_fit1, tmp_fit2 = nmt_test(rnnsearch_path_orig, wmt_data_path, GPU_ID, translate_param1_path, translate_param2_path, group_dict)
       print("The original model inspection")
       print("{} parameters in total.".format(total_param))
       print('validatoin => acc (%.4f), ppl (%.4f)' % ( tmp_fit1[1], tmp_fit1[0]))
       print('testing => bleu (%.4f), ppl (%.4f)' % (tmp_fit2[1]*100, tmp_fit2[0]))
     elif args.status == 'pruned':
       print("The pruned model inspection")
       total_param, sparsity, tmp_fit1, tmp_fit2 = nmt_test(rnnsearch_path_pruned, wmt_data_path, GPU_ID, translate_param1_path, translate_param2_path, group_dict)
       print("{} parameters in total.".format(total_param))
       print("{} parameters are pruned.".format(sparsity * total_param))
       print("sparsity {}%".format(100. * sparsity))
       print('validatoin => acc (%.4f), ppl (%.4f)' % ( tmp_fit1[1], tmp_fit1[0]))
       print('testing => bleu (%.4f), ppl (%.4f)' % (tmp_fit2[1]*100, tmp_fit2[0]))
   elif args.model == 'luongnet':
     print("========MNT Model: LuongNet==========")
     from luongnet_layer_group import group_dict 
     if args.status == 'orig':
       total_param, sparsity, tmp_fit1, tmp_fit2 = nmt_test(luongnet_path_orig, wmt_data_path, GPU_ID, translate_param1_path, translate_param2_path, group_dict)
       print("The original model inspection")
       print("{} parameters in total.".format(total_param))
       print('validatoin => acc (%.4f), ppl (%.4f)' % ( tmp_fit1[1], tmp_fit1[0]))
       print('testing => bleu (%.4f), ppl (%.4f)' % (tmp_fit2[1]*100, tmp_fit2[0]))
     elif args.status == 'pruned':
       print("The pruned model inspection")
       total_param, sparsity, tmp_fit1, tmp_fit2 = nmt_test(luongnet_path_pruned, wmt_data_path, GPU_ID, translate_param1_path, translate_param2_path, group_dict)
       print("{} parameters in total.".format(total_param))
       print("{} parameters are pruned.".format(sparsity * total_param))
       print("sparsity {}%".format(100. * sparsity))
       print('validatoin => acc (%.4f), ppl (%.4f)' % ( tmp_fit1[1], tmp_fit1[0]))
       print('testing => bleu (%.4f), ppl (%.4f)' % (tmp_fit2[1]*100, tmp_fit2[0]))
   else:
     print("ERROR OPTION OF MODEL")
