# CPGA

The dataset and code of the paper "CPGA: Coding Priors-Guided Aggregation Network for Compressed Video Quality Enhancement".

# Requirements

CUDA==11.3 Python==3.7 
## Environment
```python
conda create -n cpga python=3.7 -y && conda activate cpga

git clone --depth=1 https://github.com/VQE-CPGA/CPGA && cd VQE-CPGA/CPGA/

# given CUDA 11.3
python -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
```
## DCNv2
```python
cd ops/dcn/
bash build.sh
```
Check if DCNv2 work(optional)
```python
python simple_check.py
```
## VCP dataset
Download raw and compressed videos [![BaiduPan]](https://blog.csdn.net/A33280000f/article/details/115836658)

# Train
```python
python train_LD_QP22.py --opt_path ./config/option_CVQE_V3_7_6_cvpd_LDB_37.yml
```
# Test
```python
python test_LD_QP22.py --opt_path ./config/option_CVQE_V3_7_6_cvpd_LDB_37.yml
```
# Citation
```python
@inproceedings{2024qiang_cpga,
  title={Coding Priors-Guided Aggregation Network for Compressed Video Quality Enhancement},
  author={Qiang Zhu, Jinhua Hao, Yukang Ding, Yu Liu, Qiao Mo, Ming Sun, Chao Zhou, Shuyuan Zhu},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition 2024},
  volume={},
  number={},
  pages={},
  year={2024}
}
```
