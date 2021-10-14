# Image-to-Image Mixer for Image Reconstruction

This repository containes the code to reproduce the results and figures of the paper: Image-to-Image MLP-mixer for Image Reconstruction (add paper link later). 

## Data Sets 

We used 3 data sets in the paper. The first one is a mini version of ImageNet and can be downloaded from [here](https://www.kaggle.com/ifigotin/imagenetmini-1000). The second one is the Smart Phone Image Denoising Dataset [(SIDD)](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php). Specifically, we used the SIDD-Small Dataset with sRGB images only, which can be directly downloaded [here](https://competitions.codalab.org/my/datasets/download/a26784fe-cf33-48c2-b61f-94b299dbc0f2). 

The data needs to be preprocessed before using it for training. A notebook called "prepare_data.ipynb" exists in each of the folders "ImageNet" and "SIDD" and is responsible for processing the data such that it's easily used for training, and then saving the new processed data. The file "config.py" in both folders should be edited with the path to the original dataset and the path in which the processed data should be saved.  

The third one is the fastMRI knee data (write more details here).

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
* (add more for vit)

## Usage

## Acknowledgements

## Citation

## License
