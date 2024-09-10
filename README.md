# 📋Lagrange-Duality-and-CMAformer

Lagrange Duality and Compound Multi-Attention Transformer (CMAformer) for Semi-Supervised Medical Image Segmentation

[Fuchen Zheng](https://lzeeorno.github.io/), Quanjun Li, Weixuan Li,  [Xuhang Chen](https://cxh.netlify.app/), Yihang Dong, Guoheng Huang, [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/) 📮and [Shoujun Zhou](https://people.ucas.edu.cn/~sjzhou?language=en) 📮( 📮 Corresponding authors)

**University of Macau, SIAT CAS, Guangdong University of Technology**


## 🚧 Installation 
Requirements: `Ubuntu 20.04`

1. Create a virtual environment: `conda create -n your_environment python=3.8 -y` and `conda activate your_environment `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) :`pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118`
Or you can use Tsinghua Source for installation
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```
3. `pip install tqdm scikit-learn albumentations==1.0.3 pandas einops axial_attention`


5. `python train_lits2017_png.py`


## 1. Prepare the dataset

### LiTS2017 datasets
- The LiTS2017 datasets can be downloaded here: {[LiTS2017](https://competitions.codalab.org/competitions/17094)}. 

- After downloading the datasets, you should run ./data_prepare/preprocess_lits2017_png.py to convert .nii files into .png files for training.(Save the downloaded LiTS2017 datasets in the data folder in the following format.)

- './data_prepare/'
  - preprocess_lits2017_png.py
- './data/'
  - LITS2017
    - ct
      - .nii
    - label
      - .nii
  - trainImage_lits2017_png
      - .png
  - trainImage_lits2017_png
      - .png

## 2. Prepare the pre_trained weights
- The weights of the pre-trained CMAformer could be downloaded [here](). After that, the pre-trained weights should be stored in './pretrained_weights/'.
- To use pre-trained file, you should change 2 places in './train/train_lits2017_png.py'
  - 1. Change 'default=True' in 'parser.add_argument('--pretrained', default=False, type=str2bool)'
  - 2. Change 'pretrained_path= "./your_pretrained_file_path"' after 'if args.model_name == 'CMAformer':'

## 3. Train the CMA-Former
```bash
cd ./train/
python train_lits2017_png.py 
```

## 4. Obtain the outputs
- After trianing, you could obtain the results in './trained_models/LiTS_CMAformer/'

  
# 🧧 Acknowledgement


# 🛎 Citation

If you find our work helpful for your research, please cite:

```bib

```

