# AMCD-Net

基于自适应多模态门控融合的工业表面缺陷检测方法。

## 环境配置
- Python 3.10
- PyTorch 2.5.0 + CUDA 12.1
- 安装依赖：`pip install -r requirements.txt`

## 数据集
下载 MVTec AD 数据集并解压到 `~/datasets/mvtec_ad`，目录结构保持原样。

## 运行
- 基线实验：`python train_patchcore_torchvision.py`
- 完整 AMCD‑Net：`python train_amcd_final.py`
- 消融实验（去掉纹理/结构/门控）：`python ablation_no_tex.py`、`ablation_no_edge.py`、`ablation_no_gate.py`
