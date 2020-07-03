#usage: 
#CUDA_VISIBLE_DEVICES=[GPU ID] python -W ignore lm_iterative_mp_pruning.py  [name_mark for logs/reocrds/saved_models] [GPU ID TO RUN ON] [MODEL TYPE: LM] [GROUPING STRATEGY FOR CONNECTIONS: simple/layer/time] [ACCURACY_CONSTRAINT] > [RECORD FILE.txt]
CUDA_VISIBLE_DEVICES=0 python -W ignore lm_sipo_pruning.py  lm_stage-wise 0 LM time 0.01 > ../running_records/lm_stage-wise_opt.txt
