# CPGA

The dataset and code of the paper "CPGA: Coding Priors-Guided Aggregation Network for Compressed Video Quality Enhancement".

# Requirements

CUDA==11.3 Python==3.7 

```python
conda create -n stdf python=3.7 -y && conda activate stdf
```
git clone --depth=1 https://github.com/ryanxingql/stdf-pytorch && cd stdf-pytorch/

# given CUDA 10.1
python -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
