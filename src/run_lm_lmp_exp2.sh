#usage: 
#CUDA_VISIBLE_DEVICES=[GPU ID] python -W ignore lm_iterative_mp_pruning.py  [name_mark for logs/reocrds/saved_models] [GPU ID TO RUN ON] [MODEL TYPE: LM] [GROUPING STRATEGY FOR CONNECTIONS: simple/layer/time] [ACCURACY_CONSTRAINT] > [RECORD FILE.txt]
CUDA_VISIBLE_DEVICES=1 python -W ignore lm_iterative_mp_pruning.py  lm_iterative_lmp 0 LM layer 0.01 > ../running_records/exp_lm_lmp_iterative_log.txt
