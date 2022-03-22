# Efficient Graph Similarity Computation - (EGSC)

This repo contains the source code and dataset for our NeurIPS 2021 paper:

[**Slow Learning and Fast Inference: Efficient Graph Similarity Computation via Knowledge Distillation**](https://papers.nips.cc/paper/2021/file/75fc093c0ee742f6dddaa13fff98f104-Paper.pdf)
<br>
2019 Conference on Neural Information Processing Systems (NeurIPS 2021)
<br>
[paper](https://papers.nips.cc/paper/2021/file/75fc093c0ee742f6dddaa13fff98f104-Paper.pdf)

<div>
    <div style="display: none;" id="egsc2021">
      <pre class="bibtex">@inproceedings{qin2021slow,
              title={Slow Learning and Fast Inference: Efficient Graph Similarity Computation via Knowledge Distillation},
              author={Qin, Can and Zhao, Handong and Wang, Lichen and Wang, Huan and Zhang, Yulun and Fu, Yun},
              booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
              year={2021}
            }
    </pre>
  </div>
  <br>
</div>

![EGSC](/Figs/our-setting.png)

## Introduction
<div>
    <br>
Graph Similarity Computation (GSC) is essential to wide-ranging graph appli- cations such as retrieval, plagiarism/anomaly detection, etc. The exact computation of graph similarity, e.g., Graph Edit Distance (GED), is an NP-hard problem that cannot be exactly solved within an adequate time given large graphs. Thanks to the strong representation power of graph neural network (GNN), a variety of GNN-based inexact methods emerged. To capture the subtle difference across graphs, the key success is designing the dense interaction with features fusion at the early stage, which, however, is a trade-off between speed and accuracy. For Slow Learning of graph similarity, this paper proposes a novel early-fusion approach by designing a co-attention-based feature fusion network on multilevel GNN features. To further improve the speed without much accuracy drop, we introduce an efficient GSC solution by distilling the knowledge from the slow early-fusion model to the student one for Fast Inference. Such a student model also enables the offline collection of individual graph embeddings, speeding up the inference time in orders. To address the instability through knowledge transfer, we decompose the dynamic joint embedding into the static pseudo individual ones for precise teacher-student alignment. The experimental analysis on the real-world datasets demonstrates the superiority of our approach over the state-of-the-art methods on both accuracy and efficiency. Particularly, we speed up the prior art by more than 10x on the benchmark AIDS data.
    <br>
</div>

## Dataset
We have used the standard dataloader, i.e., ‘GEDDataset’, directly provided in the [PyG](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/ged_dataset.html#GEDDataset).

```  AIDS700nef:  ``` https://drive.google.com/uc?export=download&id=10czBPJDEzEDI2tq7Z7mkBjLhj55F-a2z

```  LINUX:  ``` https://drive.google.com/uc?export=download&id=1nw0RRVgyLpit4V4XFQyDy0pI6wUEXSOI

```  ALKANE:  ``` https://drive.google.com/uc?export=download&id=1-LmxaWW3KulLh00YqscVEflbqr0g4cXt

```  IMDBMulti:  ``` https://drive.google.com/uc?export=download&id=12QxZ7EhYA7pJiF4cO-HuE8szhSOWcfST


<p align="justify">
The code takes pairs of graphs for training from an input folder where each pair of graph is stored as a JSON. Pairs of graphs used for testing are also stored as JSON files. Every node id and node label has to be indexed from 0. Keys of dictionaries are stored strings in order to make JSON serialization possible.</p>

Every JSON file has the following key-value structure:

```javascript
{"graph_1": [[0, 1], [1, 2], [2, 3], [3, 4]],
 "graph_2":  [[0, 1], [1, 2], [1, 3], [3, 4], [2, 4]],
 "labels_1": [2, 2, 2, 2],
 "labels_2": [2, 3, 2, 2, 2],
 "ged": 1}
```
<p align="justify">
The **graph_1** and **graph_2** keys have edge list values which descibe the connectivity structure. Similarly, the **labels_1**  and **labels_2** keys have labels for each node which are stored as list - positions in the list correspond to node identifiers. The **ged** key has an integer value which is the raw graph edit distance for the pair of graphs.</p>

## Requirements
The codebase is implemented in Python 3.6.12. package versions used for development are just below.
```
matplotlib        3.3.4
networkx          2.4
numpy             1.19.5
pandas            1.1.2
scikit-learn      0.23.2
scipy             1.4.1
texttable         1.6.3
torch             1.6.0
torch-cluster     1.5.9
torch-geometric   1.7.0
torch-scatter     2.0.6
torch-sparse      0.6.9
tqdm              4.60.0
```

## File Structure
```
.
├── README.md
├── LICENSE                            
├── EGSC-T
│   ├── src
│   │    ├── egsc.py 
│   │    ├── layers.py
│   │    ├── main.py
│   │    ├── model.py
│   │    ├── parser.py        
│   │    └── utils.py                             
│   ├── README.md                      
│   └── train.sh
├── EGSC-KD
│   ├── src
│   │    ├── egsc_kd.py 
│   │    ├── egsc_nonkd.py 
│   │    ├── layers.py
│   │    ├── main_kd.py
│   │    ├── main_nonkd.py
│   │    ├── model_kd.py
│   │    ├── parser.py    
│   │    ├── trans_modules.py    
│   │    └── utils.py                             
│   ├── README.md  
│   ├── train_kd.md                     
│   └── train_nonkd.sh 
├── Checkpoints
│   ├── G_EarlyFusion_Disentangle_LINUX_gin_checkpoint.pth
│   ├── G_EarlyFusion_Disentangle_IMDBMulti_gin_checkpoint.pth
│   ├── G_EarlyFusion_Disentangle_ALKANE_gin_checkpoint.pth
│   └── G_EarlyFusion_Disentangle_AIDS700nef_gin_checkpoint.pth                         
└── GSC_datasets
    ├── AIDS700nef
    ├── ALKANE
    ├── IMDBMulti
    └── LINUX
```

## To Do
- [x] GED Datasets Processing
- [x] Teacher Model Training
- [x] Student Model Training
- [x] Knowledge Distillation
- [ ] Online Inference

## Acknowledgement
We would like to thank the [SimGNN](https://github.com/benedekrozemberczki/SimGNN) and [Extended-SimGNN](https://github.com/gospodima/Extended-SimGNN) which we used for this implementation.

## Hint
On some datasets, the results are not quite stable. We suggest to run multiple times to report the avarage one.
