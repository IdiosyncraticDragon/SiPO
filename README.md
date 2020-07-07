# Stage-wise Grouping for Recurrent Neural Networks Pruning



## Introduction

This code is the implementation of the paper "Stage-wise Grouping for Recurrent Neural Networks Pruning". In this project, we will provide the download links for preprocessed data, original model and the pruned model. Then, we will provide scripts to run the pruned model to inspect the performance of pruning w.r.t to the number of connections pruned and the performance change on the dataset. Finally, we will provide a demonstration of how to prune the neural network models by the project, using the language model as example (The scripts for NMT pruning is also provided, but it need nearly a week to run the script, so that we do not suggest the user to run it for just a demonstration).

## Environment setup

### install by anaconda

1. Firstly, the user needs to install anaconda.
2. Then, check the `requirments.yaml` and `requirments.txt` in the project folder.
3. Install the python environment by:

```bash
$> conda env create -f requirements.yaml -n SiPO-env
$> conda activate SiPO-env
$> pip install -i  https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

4. The environment is installed.

***OR***

you can run the `setup.sh`.



### build by docker

**To Be Finished.**



## Data download

1. Penn Treebank Dataset (PTB) for language model (LM)

   - raw data download: 

     https://catalog.ldc.upenn.edu/LDC95T7

     https://catalog.ldc.upenn.edu/LDC99T42

   - data used in this project:  Check the folder `data/penn` under the project.

2. WMT14 EN-DE for neural machine translation (NMT)

   - raw data download: http://statmt.org/wmt14

   - Pre-processed data used in this project:  [Web URL](https://pan.baidu.com/s/1tjknRHSZGTu4TiH5wrsAGw)

   The structure of folder should be:

   ```
   /SiPO
   
   |--/SiPO/data
   
   |----/SiPO/data
   
   |------/SiPO/data/penn
   
   |--------/SiPO/data/penn/train.txt
   
   |--------/SiPO/data/penn/valid.txt
   
   |--------/SiPO/data/penn/test.txt
   
   |----/SiPO/wmt14
   
   |------/SiPO/wmt14/len50_pywmt14.train.pt
   
   |------/SiPO/wmt14/len50_pywmt14.valid.pt
   
   |------/SiPO/wmt14/len50_pywmt14.vocab.pt
   
   |------/SiPO/wmt14/en-test.txt
   
   |------/SiPO/wmt14/de-test.txt
   ```
   
   

## Model download

1. Original models for LM and NMT

   [Web URL](https://pan.baidu.com/s/1Z5H5SX64-aYHug3uho59CA)

2. Pruned models of LM and NMT

   [Web URL](https://pan.baidu.com/s/1A65qMcYm4VFr8xJ422oArA)

**Note:** put the downloaded models into the folder `model` under this project.



### Inspect the pruned models

**Note:** The pruning is implemented by setting the pruned connections' weight value to zero, so that the direct storage of the pruned models is the same as the original models. In this section, the user can use provided script to find out how many connections are pruned in the pruned model and check the performances of the pruned model on the dataset.

- Language models:

```bash
$> cd utils
$> chmod +x lm_run_scripts.sh
$> ./lm_run_scripts.sh
```

If the environment is set successfully, the data and models are download and placed to the `data` folder and `model` folder, the results will be:

```bash
$> ./lm_run_scripts.sh
========Language Model==========
4653200 paramters in totall
Valid ppl: 120.40824300174098
Test ppl: 117.21596047584426
Valid acc: 23.510032537960953\%
Test acc: 23.00376076671115
========Language Model==========
/home2/lgy/installed/anacoda3/envs/SiPO-env/lib/python3.6/site-packages/torch/serialization.py:325: SourceChangeWarning: source code of class 'masked_networkLM.MaskedModel' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
4653200 paramters in totall
4499874 paramters prund
sparsity: 0.9670493423880341\%
Valid ppl: 133.71846670654708
Test ppl: 128.7818224302926
Valid acc: 23.636117136659436\%
Test acc: 23.288851146427273

```

**Explanation:**

The "========Language Model==========" is the results split bar, which indicate a evaluation for one model.

The first "========Language Model==========" indicates the evaluation of the original LM model. The following results indicate the evaluation of the pruned LM model.

- RNNSearch:

```bash
$> cd utils
$> chmod +x rnnsearch_run_scripts.sh
$> ./rnnsearch_run_scripts.sh
```

If the environment is set successfully, the data and models are download and placed to the `data` folder and `model` folder, the results will be:

```bash
========NMT Model: RNNSearch==========
Loading model parameters.
Loading model parameters.
The original model inspection
137804724 parameters in total.
validatoin => acc (55.9544), ppl (9.1475)
testing => bleu (18.2537), ppl (1.6292)
========NMT Model: RNNSearch==========
The pruned model inspection
Loading model parameters.
Loading model parameters.
137804724 parameters in total.
116952767.0 parameters are pruned.
sparsity 84.86847446536014%
validatoin => acc (55.9262), ppl (9.3694)
testing => bleu (17.9691), ppl (1.6360)

```



- LuongNet:

```bash
$> cd utils
$> chmod +x luongnet_run_scripts.sh
$> ./luongnet_run_scripts.sh
```

If the environment is set successfully, the data and models are download and placed to the `data` folder and `model` folder, the results will be:

```bash
========MNT Model: LuongNet==========
Loading model parameters.
Loading model parameters.
The original model inspection
221124004 parameters in total.
validatoin => acc (58.3489), ppl (7.5092)
testing => bleu (19.4961), ppl (1.5681)
========MNT Model: LuongNet==========
The pruned model inspection
Loading model parameters.
Loading model parameters.
221124004 parameters in total.
191360406.0 parameters are pruned.
sparsity 86.5398611360167%
validatoin => acc (58.3312), ppl (7.6990)
testing => bleu (19.2241), ppl (1.5567)

```



## Experiments for Language model

Organized by the connection grouping strategies.

- **All-in-one Strategy**

  1. pruning without retraining:

  ```bash
  $> conda activate SiPO-env
  $> cd SiPO/src
  $> chmod +x run_lm_mp_exp1.sh
  $> ./run_lm_mp_exp1.sh
  ```

  When the running is finished

     - the running record is saved under  `SiPO/running_records/lm_simple_grid.txt`
     - the pruned models in each iteration can be found under the `__C.LM_MODEL_PATH` defined in `SiPO/src/config.py`

  2. iterative pruning and retraining:

  ```bash
  $> conda activate SiPO-env
  $> cd SiPO/src
  $> chmod +x run_lm_mp_exp2.sh
  $> ./run_lm_mp_exp2.sh
  ```

  When the running is finished

     - the running record is saved under  `SiPO/running_records/exp_lm_mp_iterative_log.txt`

     - the retrained models in each iteration can be found under the `__C.LM_MODEL_TMP_FOLDER` defined in `SiPO/src/config.py`

     - the pruned models in each iteration can be found under the `__C.LM_MODEL_PATH` defined in `SiPO/src/config.py`

       **Note:** The best model is the retrained one whose  accuracy can be recovered by retraining. For example, if the model can not be retrained to recover its accuracy in the 8-th iteration, then the best model is the retrained on in the 7-th iteration.

- **Layer-wise Strategy**

1. pruning without retraining:

```bash
$> conda activate SiPO-env
$> cd SiPO/src
$> chmod +x run_lm_lmp_exp1.sh
$> ./run_lm_lmp_exp1.sh
```

When the running is finished

   - the running record is saved under  `SiPO/running_records/lm_layer_grid.txt`
   - the pruned models in each iteration can be found under the `__C.LM_MODEL_PATH` defined in `SiPO/src/config.py`

2. iterative pruning and retraining:

```bash
$> conda activate SiPO-env
$> cd SiPO/src
$> chmod +x run_lm_lmp_exp2.sh
$> ./run_lm_lmp_exp2.sh
```

When the running is finished

   - the running record is saved under  `SiPO/running_records/exp_lm_lmp_iterative_log.txt`

   - the retrained models in each iteration can be found under the `__C.LM_MODEL_TMP_FOLDER` defined in `SiPO/src/config.py`

   - the pruned models in each iteration can be found under the `__C.LM_MODEL_PATH` defined in `SiPO/src/config.py`

     **Note:** The best model is the retrained one whose  accuracy can be recovered by retraining. For example, if the model can not be retrained to recover its accuracy in the 8-th iteration, then the best model is the retrained on in the 7-th iteration.

- **Stage-wise Strategy**

1. pruning without retraining:

```bash
$> conda activate SiPO-env
$> cd SiPO/src
$> chmod +x run_lm_sipo_exp1.sh
$> ./run_lm_sipo_exp1.sh
```

When the running is finished

   - the running record is saved under  `SiPO/running_records/lm_stage-wise_opt.txt`
   - the pruned models in each iteration can be found under the `__C.LM_MODEL_PATH` defined in `SiPO/src/config.py`

2. iterative pruning and retraining:

```bash
$> conda activate SiPO-env
$> cd SiPO/src
$> chmod +x run_lm_sipo_exp2.sh
$> ./run_lm_sipo_exp2.sh
```

When the running is finished

   - the running record is saved under  `SiPO/running_records/exp_lm_sipo_iterative_log.txt`

   - the retrained models in each iteration can be found under the `__C.LM_MODEL_TMP_FOLDER` defined in `SiPO/src/config.py`

   - the pruned models in each iteration can be found under the `__C.LM_MODEL_PATH` defined in `SiPO/src/config.py`

     **Note:** The best model is the retrained one whose  accuracy can be recovered by retraining. For example, if the model can not be retrained to recover its accuracy in the 8-th iteration, then the best model is the retrained on in the 7-th iteration.