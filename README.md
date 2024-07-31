# Thesis
Unsupervised Super Resolution for Sentinel-2 satellite imagery
------------------------------------------------------------------------------------------------------------------
In this work three unsupervised deep learning models were utilized for Super Resolving satellite imagery obtained from Sentinel-2 constellation. 

## 1. Deep Image Prior (DIP) 
The original implementation of this model was proposed by [Ulyanov et al.](https://github.com/DmitryUlyanov/deep-image-prior) 

### Requirments 
- python == 3.8
- earthpy
- numpy
- pytorch
- matplotlib
- scikit-image
- gdal
- rasterio
- jupyter notebook

## 2. Zero-Shot Super Resolution (ΖSSR)
The original implementation of this model was proposed by [Shocher et al.](https://github.com/assafshocher/ZSSR) 

### Requirments 
- python == 2.7
- numpy
- tensorflow
- matplotlib
- scikit-image
- opencv-python
- imageio

## 3. Degradation-Aware Super Resolution (DASR) 
The original implementation of this modes was proposed by [Wang et al.](https://github.com/The-Learning-And-Vision-Atelier-LAVA/DASR)

### Requirments
- Python 3.6
- PyTorch == 1.1.0
- numpy
- skimage
- imageio
- matplotlib
- cv2

### Train
#### 1. Prepare training data 

1.1 Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  dataset and the [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset.

1.2 Combine the HR images from these two datasets in `your_data_path/DF2K/HR` to build the DF2K dataset. 

### 2. Begin to train
Run `./main.sh` to train on the DF2K dataset. Please update `dir_data` in the bash file as `your_data_path`.


### Test
#### 1. Prepare test data 
Download [benchmark datasets](https://github.com/xinntao/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) (e.g., Set5, Set14 and other test sets) and prepare HR/LR images in `your_data_path/benchmark`.


#### 2. Begin to test
Run `./test.sh` to test on benchmark datasets. Please update `dir_data` in the bash file as `your_data_path`.


### Quick Test on An LR Image
Run `./quick_test.sh` to test on an LR image. Please update `img_dir` in the bash file as `your_img_path`.

