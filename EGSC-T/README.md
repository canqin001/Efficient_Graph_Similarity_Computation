# Efficient Graph Similarity Computation - Teacher Model
![GSCG-T](../Figs/Teacher-Net.png)
<b> Overview of early-feature fusion network (Teacher Net) which is composed of a featureencoder and a regression head as the whole.  Within the the feature encoder, there are multiplecomponents including GIN as the backbone, the Embedding Fusion Network (EFN) and graphpooling. The regression head is a MLP which projects the joint embedding into the desired similarity. </b>

## Train & Test
If you run the experiment on AIDS, then
```
python src/main.py --dataset AIDS700nef --gnn-operator gin --epochs 6000 --batch-size 128 --learning-rate 0.001
```
If you run the experiment on LINUX, then
```
python src/main.py --dataset LINUX --gnn-operator gin --epochs 6000 --batch-size 128 --learning-rate 0.001
```
If you run the experiment on IMDB, then
```
python src/main.py --dataset IMDBMulti --gnn-operator gin --epochs 6000 --batch-size 128 --learning-rate 0.001
```
If you run the experiment on ALKANE, then
```
python src/main.py --dataset ALKANE --gnn-operator gin --epochs 6000 --batch-size 128 --learning-rate 0.001
```
, or run experiments on all scenarios.
```
bash train.sh
```
