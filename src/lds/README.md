This folder contains the comparative test code related to the LDS pruning algorithm. The algorithm is implemented with reference to the algorithm description of the original paper.

```
Chatzikonstantinou, C., Konstantinidis, D., Dimitropoulos, K., & Daras, P. (2021). Recurrent neural network pruning using dynamical systems and iterative fine-tuning. Neural Networks, 143, 475-488.
```



#### Related files

```python
lm_lds_oneshot.py # oneshot pruning for languages model
lm_lds_iterative.py # iterative pruning for languages model
nmt_lds_oneshot.py  # oneshot pruning for opennmt model
nmt_lds_iterative.py  # iterative pruning for opennmt model
```



#### Notice:

+ The relevant code should be placed in the src folder for execution. Here, a separate folder is used for the convenience of sorting.
+ Since we tried multiple sets of parameters in the experiment, the relevant parameters may be different from the real experiment, you need to adjust the relevant parameters yourself
+ In the iterative experiment file, we only show one iterative process, you can control your own number of iterations as needed
+ There is a certain gap between our results and the original paper. In the absence of the original paper author's email reply, the original paper code, and the original paper description also have some ambiguity, we implemented the algorithm step by step as much as possible according to the paper description, and carried out multiple proofreading.
+ For some newly added metrics, we use the open source toolkit nlg-eval for pruning calculations. In order to preserve environment consistency, nlg-eval related code is not included here. For specific usage, please refer to the description of nlg-eval (https://github.com/Maluuba/nlg-eval).



