# DIGNNs

This repository provides a reference implementation of **DIGNN** as described in the paper "**[Implicit Graph Neural Diffusion Based on Constrained Dirichlet Energy Minimization](https://arxiv.org/pdf/2308.03306.pdf)**" which has been presented at NeurIPS 2023 New Frontiers in Graph Learning Workshop.


## Requirements

* Install [**PyTorch >= 1.7.0**](https://pytorch.org/get-started/locally/)
* Install [**PyTorch Geometric >= 1.7.0**](https://github.com/rusty1s/pytorch_geometric#installation)

## Run Experiments 
We provide some examples for running experiments for different tasks on different datasets:
### Node classification 
```
cd nodeclassification
```

For chameleon and squirrel datasets,
```
python main.py --input chameleon --model Neural --mu 2.2 --preprocess adj --max_iter 10 --dropout 0.5 --lr 0.01 --weight_decay 0
```

For PPI dataset,
```
python main_ppi.py --model Neural --dropout 0.1 --epoch 1000 --num_hid 512 --lr 0.01 --mu 2 --weight_decay 0 --max_iter 10
```

### Graph classification
```
cd graphclassification
```
```
python main.py --input MUTAG --model Neural --mu 1.25 --max_iter 20 --num_hid 128 --lr 0.001 --weight_decay 0 --epochs 1000 
```

## Citing
If you find *DIGNN* useful in your research, please cite our paper:
```
@article{DBLP:journals/corr/abs-2308-03306,
  author       = {Guoji Fu and
                  Mohammed Haroon Dupty and
                  Yanfei Dong and
                  Lee Wee Sun},
  title        = {Implicit Graph Neural Diffusion Based on Constrained Dirichlet Energy
                  Minimization},
  journal      = {CoRR},
  volume       = {abs/2308.03306},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2308.03306},
  doi          = {10.48550/ARXIV.2308.03306},
  eprinttype    = {arXiv},
  eprint       = {2308.03306},
  timestamp    = {Mon, 21 Aug 2023 17:38:10 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2308-03306.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
