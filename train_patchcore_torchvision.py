import torch
import torchvision.models as tv_models
from torchvision.transforms import Resize
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# 自定义变换
class ResizeOnly:
    def __init__(self, size):
        self.resize = Resize(size, antialias=True)
    def __call__(self, image, mask=None):
        image = self.resize(image)
        return (image, mask) if mask is not None else image

transform = ResizeOnly((128, 128))

# 数据模块
dm = MVTecAD(
    root="/home/lili/datasets/mvtec_ad",
    category="bottle",
    train_batch_size=4,
    eval_batch_size=4,
    train_augmentations=transform,
    val_augmentations=transform,
    test_augmentations=transform,
)

# 用 torchvision 的 resnet18，pretrained=True 会从 pytorch.org 下载权重
backbone = tv_models.resnet18(pretrained=True)

# 创建 Patchcore 模型，直接传入 backbone
model = Patchcore(backbone=backbone)

# 训练 + 测试
engine = Engine()
engine.train(model=model, datamodule=dm)
results = engine.test(model=model, datamodule=dm)

print("\n========== 最终测试结果 ==========")
for k, v in results[0].items():
    print(f"{k}: {v}")
