from model_inspect import *
import argparse

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="inspect the status of models") 
   parser.add_argument('--model', choices=['lm','rnnsearch', 'luongnet'], default='lm')
   parser.add_argument('--status', choices=['orig','pruned'], default='orig')
   #parser.add_argument('--opt', choices=['num_param','sparsity', 'valid_ppl', 'test_ppl', 'test_bleu', 'valid_acc', 'test_acc'], default='num_param')
   args = parser.parse_args()

   lm_data_path = '../data/penn/'
   lm_model_path_orig = '../model/original_model/language_model/lm_model_orignal.pt'
   lm_model_path_pruned = '../model/pruned_model/language_model/lm_model_pruned.pt'

   # procedure for language models
   if args.model == 'lm':
     print("========Language Model==========")
     corpus, _, val_data, test_data = lm_data_load(lm_data_path)
     if args.status == 'orig':
         model = lm_model_load(lm_model_path_orig)
         total_num = lm_original_param_num(model)
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
         print("{} paramters in totall".format(total_num))
         print("{} paramters prund".format(int(sparsity * total_num)))
         print("sparsity: {}\%".format(sparsity))
         val_ppl, val_acc = lm_evaluate(model.masked_model, corpus, val_data)
         test_ppl, test_acc = lm_evaluate(model.masked_model, corpus, test_data)
         print("Valid ppl: {}".format(val_ppl))
         print("Test ppl: {}".format(test_ppl))
         print("Valid acc: {}\%".format(100. * val_acc))
         print("Test acc: {}".format(100. * test_acc))
     else:
         print("ERROR OPTION")

