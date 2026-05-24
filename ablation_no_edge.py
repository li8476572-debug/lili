import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tv_models, torchvision.transforms as T
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

class ResizeOnly:
    def __init__(self, size):
        self.resize = T.Resize(size, antialias=True)
    def __call__(self, image, mask=None):
        return (self.resize(image), mask) if mask is not None else image

class TextureBranch(nn.Module):
    def __init__(self, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class GatedFusion(nn.Module):
    def __init__(self, v_dim, t_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(v_dim+t_dim, 128), nn.ReLU(), nn.Linear(128, 2)
        )
    def forward(self, fv, ft):
        gv = fv.mean(dim=[2,3]); gt = ft.mean(dim=[2,3])
        w = F.softmax(self.fc(torch.cat([gv,gt],1)), 1)
        return w[:,0:1,None,None]*fv + w[:,1:2,None,None]*ft

class MultiModalBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = tv_models.resnet18(pretrained=True)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.tex = TextureBranch(64)
        self.tex_proj = nn.Conv2d(64, 128, 1)
        self.gate = GatedFusion(128, 128)

    def forward(self, x):
        v0 = self.stem(x)
        v1 = self.layer1(v0)
        v2 = self.layer2(v1)
        t = self.tex(x)
        t = F.interpolate(t, size=v2.shape[2:], mode='bilinear')
        t = self.tex_proj(t)
        fused = self.gate(v2, t)          # 无结构分支
        v3 = self.layer3(v2)
        return [fused, v3]

transform = ResizeOnly((128,128))
dm = MVTecAD(root="/home/lili/datasets/mvtec_ad", category="bottle",
              train_batch_size=4, eval_batch_size=4,
              train_augmentations=transform, val_augmentations=transform, test_augmentations=transform)
model = Patchcore(backbone=MultiModalBackbone())
engine = Engine()
engine.train(model=model, datamodule=dm)
results = engine.test(model=model, datamodule=dm)
print("\n========== 消融：去掉结构分支 ==========")
for k, v in results[0].items():
    print(f"{k}: {v}")
