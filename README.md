# RRSR:Reciprocal Reference-based Image Super-Resolution with Progressive Feature Alignment and Selection

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>


[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790637.pdf)]
[[ArXiv](https://arxiv.org/abs/2211.04203)]

## Installation
Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/wdmwhh/C2M_RAB-CIFT-RTRR.git
cd C2M_RAB-CIFT-RTRR
conda env create -f environment.yml
source activate mmlab113
```

## Inference

### Test on CUFED5
```
python mmsr/test.py -opt options/test/test_mse.yml
```
The results will be saved in ./result

## Training

### Train on CUFED5
```
python mmsr/train.py -opt options/train/stage3_restoration_mse.yml
```
The results will be saved in ./experiments
