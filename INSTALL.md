## Installation

### Rui

#### Environment:

- coda env: py310matsegnerf
- cuda 12.1
- PyTorch Nightly: `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121`

#### [Detectron](https://detectron2.readthedocs.io/en/latest/tutorials/install.html):

``` bash
conda install -c conda-forge gcc # see notes below
conda install -c conda-forge gxx_linux-64
conda install -c conda-forge opencv
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Notes installing gcc: 

- gcc compatibility with cuda versions: https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version; 
- to install specific version of gcc with conda: https://stackoverflow.com/questions/47955200/specific-package-version-with-conda-forge



### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

#### Building on another system
To build on a system that does not have a GPU device but provide the drivers:
```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### Example conda environment setup
```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```
