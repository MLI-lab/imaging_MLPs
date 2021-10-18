# Image-to-Image Mixer for Image Reconstruction

This repository containes the code to reproduce the results and figures of the paper: Image-to-Image MLP-mixer for Image Reconstruction (add paper link later). 

## Data Sets 

We used 3 data sets in the paper. The first one is a mini version of ImageNet and can be downloaded from [here](https://www.kaggle.com/ifigotin/imagenetmini-1000). The second one is the Smart Phone Image Denoising Dataset [(SIDD)](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php). Specifically, we used the SIDD-Small Dataset with sRGB images only, which can be directly downloaded [here](https://competitions.codalab.org/my/datasets/download/a26784fe-cf33-48c2-b61f-94b299dbc0f2). 

The data needs to be preprocessed before using it for training. A notebook called "prepare_data.ipynb" exists in each of the folders "ImageNet" and "SIDD" and is responsible for processing the data such that it's easily used for training, and then saving the new processed data. The file "config.py" in both folders should be edited with the path to the original dataset and the path in which the processed data should be saved.  

The third one is the fastMRI knee dataset and can be downloaded from the [official website](https://fastmri.med.nyu.edu/).

## Installation

The code is written entirely in python and relies mainly on Pytorch. It has been tested with the following:
* Operating System: Ubuntu 18.04.4
* Cuda: 11.0
* Python: 3.8.3
* Pytorch: 1.7.1
* Tensorflow: 2.3.0
* Jupyter Notebook: 6.3.0
* Numpy: 1.19.2
* Matplotlib: 3.3.4
* PIL: 7.2.0
* Skimage: 0.18.1
* BM3D: 3.0.9
* Einops: 0.3.0
* timm: 0.3.2
* runstats: 2.0.0
* h5py: 2.10.0

## Usage

### Denoising:

The folders "ImageNet" and "SIDD" contain the denoising experiments. In the file "config.py", the path to store the trained networks should be entered. If you wish to use tensorboard while training, you can enter the path to store its logs in "config.py". After running "prepare_data.ipynb", the processed dataset will be saved. The notebook "train.ipynb" can now be run for training. At the beginning of the notebook, the hyperparameters for the optimizer can be configured and the network to be trained can be chosen. Once training is over, the notebook "evaluate.ipynb" can be run to display the denoising performance of the trained network. All networks used in the paper are in the folder "networks". Finally, the notebook "BM3D.ipynb" can be used to evaluate the BM3D algorithm.

### Inductive Bias:

The code to replicate the figures from the inductive bias section is in the folder "untrained". The car image used in these experiments is included in the folder. 

### Compressive Sensing: 

The code to replicate the results from the compressive sensing section is in the folder "compressed_sensing". Before running the notebook "training_and_validation.ipynb", please adjust the data path and select the model.

## Acknowledgements
* The U-Net implementation and the fastmri library are obtained from the [fastMRI repository](https://github.com/facebookresearch/fastMRI).
* Our implementation of patch merging and expanding is adapted from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet).
* A Pytorch implementation of the original [MLP-Mixer paper](https://arxiv.org/abs/2105.01601) can be found [here](https://github.com/isaaccorley/mlp-mixer-pytorch).
* Our implementation of the Vision Transformer is adapted from [ConViT](https://github.com/facebookresearch/convit)

Thank you for making your code publicly accessible.

## Citation
(add citation to the paper later)

## License
This repository is [MIT](LICENSE) licensed.
