![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-1.9.0-red.svg)
# SEMA: Semantic Distance Adversarial Learning for Text-to-Image Synthesis

Official Pytorch implementation for our paper Semantic Distance Adversarial Learning for Text-to-Image Synthesis

<img src="framework.png" width="804px" height="380px"/>

---
## Requirements
- python 3.8
- Pytorch 1.9
- transformers 4.8.1
## Installation

Clone this repo.
```
git clone https://github.com/yuanrr/SEMA
pip install -r requirements.txt
```


## Preparation
### Datasets
1. Download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco2014](http://cocodataset.org/#download) dataset and extract the images to `data/coco/images/`


## Training
  ```
  Code for training SEMA will be released soon.
  ```

## Evaluation

### Download Pretrained Model for [coco](https://pan.baidu.com/s/1xKuId0EZhpqHL0tx34rkZg) (password: guvx)


### Evaluate SEMA
We synthesize about 3w images from the test descriptions and evaluate the FID between **synthesized images** and **test images** of each dataset.


 
 
 
 
