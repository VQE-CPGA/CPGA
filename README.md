# CPGA

The dataset and code of the paper "CPGA: Coding Priors-Guided Aggregation Network for Compressed Video Quality Enhancement".

# Requirements

CUDA==11.6 

Python==3.7 

## 1.1 Environment
```python
conda create -n cpga python=3.7 -y && conda activate cpga

git clone --depth=1 https://github.com/VQE-CPGA/CPGA && cd VQE-CPGA/CPGA/

# given CUDA 11.6
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
```
## 1.2 DCNv2
```python
cd ops/dcn/
bash build.sh
```
Check if DCNv2 work (optional)
```python
python simple_check.py
```
## 1.3 VCP dataset
**Download raw and compressed videos** 

Please check [[BaiduPan]](https://blog.csdn.net/A33280000f/article/details/115836658).

**Edit YML**

You need to edit option_CPGA_#_QP#.yml file.

**Generate LMDB**

The LMDB generation for speeding up IO during training.
```python
python create_lmdb_cpga.py --opt_path option_CPGA_#_QP#.yml
```
Finally, the VCP dataset root will be sym-linked to the folder ./data/ automatically.

## 1.4 Test dataset

We use the JCT-VC testing dataset in [JCT-VC](https://ieeexplore.ieee.org/document/6317156). Download raw and compressed videos [[BaiduPan]](https://blog.csdn.net/A33280000f/article/details/115836658).

# Train
```python
python train_LD_QP22.py --opt_path ./config/option_CVQE_V3_7_6_cvpd_LDB_37.yml
```
# Test
```python
python test_LD_QP22.py --opt_path ./config/option_CVQE_V3_7_6_cvpd_LDB_37.yml
```

# Acknowledgement
Thanks for the provided released code of [STDF](https://github.com/ryanxingql/stdf-pytorch) and [MFQEv2](https://github.com/ryanxingql/mfqev2.0).

# Citation
If this repository is helpful to your research, please cite our paper:
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
@article{zhu2024deep,
  title={Deep Compressed Video Super-Resolution With Guidance of Coding Priors},
  author={Qiang Zhu, Feiyu Chen, Yu Liu, Shuyuan Zhu, Bing Zeng},
  journal={ IEEE Transactions on Broadcasting },
  volume={70},
  issue={2},
  pages={505-515},
  year={2024}
  publisher={IEEE},
  doi={10.1109/TBC.2024.3394291}
}
@article{zhu2024compressed,
  title={Compressed Video Quality Enhancement with Temporal Group Alignment and Fusion},
  author={Qiang, Zhu and Yajun, Qiu and Yu, Liu and Shuyuan, Zhu and Bing, Zeng},
  journal={IEEE Signal Processing Letters},
  year={2024},
  publisher={IEEE},
  doi={10.1109/LSP.2024.3407536}
}
```
