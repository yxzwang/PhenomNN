# Introduction
Codes for Paper: From Hypergraph Energy Functions to Hypergraph Neural Networks. *ICML 2023*. 

We show the following commands to reproduce all results in Table 1 and 2 in our paper. All hyperparameters can be found in the last of Appendix in our paper.

# Environment
python 3.9\
pytorch 1.12.1+cu113\
dgl 1.0.1+cu113
# Datasets
Dataset files should be put in the ```data``` folder. Dataset sources are from [Hypergraph convolutional networks via
equivalency between hypergraphs and undirected graphs](https://arxiv.org/abs/2203.16939) and [YOU ARE ALLSET: A MULTISET LEARNING FRAMEWORK FOR HYPERGRAPH NEURAL NETWORKS.
](https://openreview.net/pdf?id=hpBTIv2uy_E). 

For convenience, we provide a copy of datasets in this [link](https://drive.google.com/file/d/13MI8p9i1worh5SVSPYujsiXmd7x0KHLC/view?usp=sharing). 
# Reproducing Table 1. 
## PhenomNN_simple
### For cocitation and coauthorship datasets
The datasets are in ['coauthorship/dblp','coauthorship/cora','cocitation/cora','cocitation/pubmed','cocitation/citeseer']  
```
python train_faster.py --type phenomnn_s --activate_dataset coauthorship/cora  --lr 0.01 --dropout 0.7 --hidden 64 --lam0 20 --lam1 80 --alp 0.1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --sigma -1
```
```
python train_faster.py --type phenomnn_s --activate_dataset coauthorship/dblp  --lr 0.005 --dropout 0.6 --hidden 64 --lam0 100 --lam1 100 --alp 0.1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --sigma -1
```
```
python train_faster.py --type phenomnn_s --activate_dataset cocitation/cora  --lr 0.005 --dropout 0.7 --hidden 64 --lam0 0 --lam1 20 --alp 1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --sigma -1
```
```
python train_faster.py --type phenomnn_s --activate_dataset cocitation/pubmed  --lr 0.02 --dropout 0.7 --hidden 64 --lam0 0 --lam1 20 --alp 0.1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --sigma -1
```
```
python train_faster.py --type phenomnn_s --activate_dataset cocitation/citeseer  --lr 0.005 --dropout 0.7 --hidden 64 --lam0 1 --lam1 20 --alp 1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --sigma -1
```
### For NTU2012 and ModelNet40 
```
python train_faster.py --on_dataset NTU2012  --type phenomnn_s --hidden 128 --alp 0.1  --lam0 1 --lam1 1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --lr 0.001  --dropout 0.2 --sigma -1 --gvcnn_feature_structure --mvcnn_feature_structure --use_gvcnn_feature --use_mvcnn_feature --is_probH
```

```
python train_faster.py --on_dataset ModelNet40  --type phenomnn_s --hidden 128 --alp 0.05  --lam0 1 --lam1 1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --lr 0.0005  --dropout 0.4 --sigma -1 --gvcnn_feature_structure --mvcnn_feature_structure --use_gvcnn_feature --use_mvcnn_feature --is_probH
```

## PhenomNN
### For cocitation and coauthorship datasets
The datasets are in ['coauthorship/dblp','coauthorship/cora','cocitation/cora','cocitation/pubmed','cocitation/citeseer']  
```
python train_faster.py --type phenomnn --activate_dataset coauthorship/cora  --lr 0.001 --dropout 0.8 --hidden 64 --lam0 20 --lam1 100 --alp 0.1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --sigma -1
```
```
python train_faster.py --type phenomnn --activate_dataset coauthorship/dblp  --lr 0.001 --dropout 0.6 --hidden 64 --lam0 1 --lam1 1 --alp 1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --sigma -1
```
```
python train_faster.py --type phenomnn --activate_dataset cocitation/cora  --lr 0.01 --dropout 0.6 --hidden 64 --lam0 0 --lam1 20 --alp 1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --sigma -1
```
```
python train_faster.py --type phenomnn --activate_dataset cocitation/pubmed  --lr 0.01 --dropout 0.6 --hidden 64 --lam0 1 --lam1 1 --alp 1 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --sigma -1
```
```
python train_faster.py --type phenomnn_s --activate_dataset cocitation/citeseer  --lr 0.001 --dropout 0.2 --hidden 64 --lam0 20 --lam1 80 --alp 0.05 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --sigma -1
```
### For NTU2012 and ModelNet40 
```
python train_faster.py --on_dataset NTU2012  --type phenomnn_s --hidden 64 --alp 0.05  --lam0 20 --lam1 80 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --lr 0.001  --dropout 0.2 --sigma -1 --gvcnn_feature_structure --mvcnn_feature_structure --use_gvcnn_feature --use_mvcnn_feature --is_probH
```

```
python train_faster.py --on_dataset ModelNet40  --type phenomnn_s --hidden 128 --alp 0.05  --lam0 0 --lam1 20 --prop_step 16 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0 --lr 0.0005  --dropout 0.2 --sigma -1 --gvcnn_feature_structure --mvcnn_feature_structure --use_gvcnn_feature --use_mvcnn_feature --is_probH
```

# Reproducing Table 2. 

activate_dataset is the dataset name where it's in ['dhg/20newsW100', 'dhg/ModelNet40','dhg/NTU2012', 'dhg/yelp', 'dhg/walmart-trips-100', 'dhg/house-committees-100'] and feature_noise denotes is only useful for 'dhg/walmart-trips-100' and 'dhg/house-committees-100'.

## PhenomNN_simple
```
python train_faster.py  --type phenomnn_s --lr 0.01 --dropout 0.2 --hidden 256  --lam0 50 --lam1 20 --alp 0.05 --prop_step 16 --activate_dataset dhg/NTU2012 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```
```
python train_faster.py  --type phenomnn_s --lr 0.01 --dropout 0 --hidden 512  --lam0 50 --lam1 1 --alp 0.05 --prop_step 16 --activate_dataset dhg/ModelNet40 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```
```
python train_faster.py  --type phenomnn_s --lr 0.01 --dropout 0.1 --hidden 64  --lam0 1 --lam1 100 --alp 0.1 --prop_step 4 --activate_dataset dhg/yelp --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```
```
python train_faster.py --feature_noise 1 --type phenomnn_s --activate_dataset dhg/house-committees-100 --lr 0.1 --dropout 0 --hidden 512  --lam0 50 --lam1 20 --alp 0 --prop_step 16  --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```
```
python train_faster.py --feature_noise 0.6 --type phenomnn_s --activate_dataset dhg/house-committees-100 --lr 0.1 --dropout 0 --hidden 512  --lam0 1 --lam1 1 --alp 0.05 --prop_step 16  --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```

```
python train_faster.py --feature_noise 1 --type phenomnn_s --activate_dataset dhg/walmart-trips-100 --lr 0.01 --dropout 0 --hidden 256  --lam0 0 --lam1 50 --alp 1 --prop_step 16  --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```
```
python train_faster.py --feature_noise 0.6 --type phenomnn_s --activate_dataset dhg/walmart-trips-100 --lr 0.1 --dropout 0 --hidden 256  --lam0 1 --lam1 20 --alp 1 --prop_step 16  --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```
```
python train_faster.py  --type phenomnn_s --activate_dataset dhg/20newsW100 --lr 0.01 --dropout 0.2 --hidden 64  --lam0 0.1 --lam1 0 --alp 1 --prop_step 7  --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```


## PhenomNN
```
python train_faster.py  --type phenomnn_s --lr 0.01 --dropout 0.2 --hidden 256  --lam0 100 --lam1 20 --alp 0.05 --prop_step 16 --activate_dataset dhg/NTU2012 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```
```
python train_faster.py  --type phenomnn_s --lr 0.001 --dropout 0.2 --hidden 512  --lam0 0 --lam1 20 --alp 0.05 --prop_step 16 --activate_dataset dhg/ModelNet40 --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```
```
python train_faster.py  --type phenomnn_s --lr 0.01 --dropout 0.2 --hidden 64  --lam0 0 --lam1 1 --alp 0.01 --prop_step 4 --activate_dataset dhg/yelp --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```
```
python train_faster.py --feature_noise 1 --type phenomnn_s --activate_dataset dhg/house-committees-100 --lr 0.01 --dropout 0.2 --hidden 64  --lam0 50 --lam1 100 --alp 0.05 --prop_step 16  --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```
```
python train_faster.py --feature_noise 0.6 --type phenomnn_s --activate_dataset dhg/house-committees-100 --lr 0.01 --dropout 0.2 --hidden 512  --lam0 0 --lam1 1 --alp 0.05 --prop_step 16  --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```

```
python train_faster.py --feature_noise 1 --type phenomnn_s --activate_dataset dhg/walmart-trips-100 --lr 0.001 --dropout 0 --hidden 256  --lam0 0 --lam1 50 --alp 1 --prop_step 16  --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```
```
python train_faster.py --feature_noise 0.6 --type phenomnn_s --activate_dataset dhg/walmart-trips-100 --lr 0.01 --dropout 0 --hidden 256  --lam0 0 --lam1 50 --alp 1 --prop_step 16  --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```
```
python train_faster.py  --type phenomnn_s --activate_dataset dhg/20newsW100 --lr 0.01 --dropout 0 --hidden 64  --lam0 0.1 --lam1 0.1 --alp 1 --prop_step 8  --data_path ./data --save_dir ./ --print_freq 100 --epochs 1000 --gpu 0  --sigma -1
```