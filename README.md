# Query Slot Attention

This repository is the official implementation of [Unsupervised Object-centric Learning with Bi-level Optimized Query Slot Attention](http://arxiv.org/abs/2210.08990)
## Environment Setup
We provide all environment configurations in ``requirements.txt``. To install all packages, you can create a conda environment and install the packages as follows: 
```bash
conda create -n BO-QSA python=3.8
conda activate BO-QSA
pip install -r environment.txt
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
In our experiments, we used NVIDIA CUDA 11.3 on Ubuntu 20.04. Similar CUDA version should also be acceptable with corresponding version control for ``torch`` and ``torchvision``.

## Data Download
### ShapeStacks
ShapeStacks dataset is avaiable at: https://ogroth.github.io/shapestacks/ 
```bash
# Download compressed dataset
$ cd data/ShapeStacks
$ wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-mjcf.tar.gz
$ wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-meta.tar.gz
$ wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-rgb.tar.gz
$ wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-iseg.tar.gz
# Uncompress files
$ tar xvzf shapestacks-meta.tar.gz
$ tar xvzf shapestacks-mjcf.tar.gz
$ tar xvzf shapestacks-rgb.tar.gz
$ tar xvzf shapestacks-iseg.tar.gz
```
### ObjectsRoom
ObjectsRoom dataset is avaiable at: https://console.cloud.google.com/storage/browser/multi-object-datasets;tab=objects?prefix=&forceOnObjectsSortingFiltering=false, provided by [Multi-Object Datasets](https://github.com/deepmind/multi_object_datasets)
```bash
# Download compressed dataset
$ cd data/ObjectsRoom
$ gsutil -m cp -r \
  "gs://multi-object-datasets/objects_room" \
  .
```
Before you start training, you need to run ``objectsroom_process.py`` to save the tfrecords dataset as a png format

### CLEVRTex
ClevrTex dataset is available at: https://www.robots.ox.ac.uk/~vgg/data/clevrtex/
### PTR
PTR dataset is available at: http://ptr.csail.mit.edu

Download the 'Training Images', 'Validation Images', 'Test Images', 'Training Annotations', 'Validation Annotations' and then uncompress them.
### Birds, Dogs, Cars
Birds, Dogs, Cars datasets are available at: https://drive.google.com/drive/folders/1zEzsKV2hOlwaNRzrEXc9oGdpTBrrVIVk, provided by [DRC](https://github.com/yuPeiyu98/DRC).

Download the 'birds.zip', 'cars.tar' and 'dogs.zip' and then uncompress them.
### Flowers
Flowers dataset is available at: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

Download the 'Dataset images', 'Image segmentations' and 'The data splits' and then uncompress them.

## Training

To train the model from scratch we provide the following model files:
 - ``train_trans_dec.py``: transformer-based model
 - ``train_mixture_dec.py``: mixture-based model
 - ``train_base_sa.py``: original slot-attention
We provide training scripts under ```scripts/train```. Please use the following command and change ``.sh`` file to the model you want to experiment with. Take the transformer-based decoder experiment on Birds as an exmaple, you can run the following:
```bash
$ cd scripts
$ cd train
$ chmod +x trans_dec_birds.sh
$ ./trans_dec_birds.sh
```
## Reload ckpts & test_only

To reload checkpoints and only run inference, we provide the following model files:
 - ``test_trans_dec.py``: transformer-based model
 - ``test_mixture_dec.py``: mixture-based model
 - ``test_base_sa.py``: original slot-attention

Similarly, we provide testing scripts under ```scripts/test```. We provide transformer-based model for real-world datasets (Birds, Dogs, Cars, Flowers) 
and mixture-based model for synthetic datasets(ShapeStacks, ObjectsRoom, ClevrTex, PTR). We provide all checkpoints at this [link](https://drive.google.com/drive/folders/10LmK9JPWsSOcezqd6eLjuzn38VdwkBUf?usp=sharing). Please use the following command and change ``.sh`` file to the model you want to experiment with:
```bash
$ cd scripts
$ cd test
$ chmod +x trans_dec_birds.sh
$ ./trans_dec_birds.sh
```

## Citation
If you find our paper and/or code helpful, please consider citing:
```
@article{jia2022egotaskqa,
    title = {Unsupervised Object-Centric Learning with Bi-Level Optimized Query Slot Attention},
    author = {Jia, Baoxiong and Liu, Yu and Huang, Siyuan},
    journal = {arXiv preprint arXiv:2210.08990},
    year = {2022}
}
```

## Acknowledgement
This code heavily used resources from [SLATE](https://github.com/singhgautam/slate), [SlotAttention](https://github.com/untitled-ai/slot_attention), [GENESISv2](https://github.com/applied-ai-lab/genesis), [DRC](https://github.com/yuPeiyu98/DRC.git), [Multi-Object Datasets](https://github.com/deepmind/multi_object_datasets), [shapestacks](https://github.com/ogroth/shapestacks). We thank the authors for open-sourcing their awesome projects.
