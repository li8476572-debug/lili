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
        self.net = nn.Sequential(nn.Conv2d(3, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU())
    def forward(self, x): return self.net(x)

class EdgeBranch(nn.Module):
    def __init__(self, out_ch=64):
        super().__init__()
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sobel_x', sx)
        self.register_buffer('sobel_y', sy)
        self.conv = nn.Sequential(nn.Conv2d(2, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU())
    def forward(self, x):
        gray = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return self.conv(torch.cat([gx, gy], dim=1))

class GatedFusion(nn.Module):
    def __init__(self, v_dim, t_dim, e_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(v_dim+t_dim+e_dim, 128), nn.ReLU(), nn.Linear(128, 3))
    def forward(self, fv, ft, fe):
        gv = fv.mean(dim=[2,3]); gt = ft.mean(dim=[2,3]); ge = fe.mean(dim=[2,3])
        w = F.softmax(self.fc(torch.cat([gv,gt,ge],1)), 1)
        return w[:,0:1,None,None]*fv + w[:,1:2,None,None]*ft + w[:,2:3,None,None]*fe

class MultiModalBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = tv_models.resnet18(pretrained=True)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.tex = TextureBranch(64)
        self.edge = EdgeBranch(64)
        self.tex_proj = nn.Conv2d(64, 128, 1)
        self.edge_proj = nn.Conv2d(64, 128, 1)
        self.gate = GatedFusion(128, 128, 128)
        self.out_channels = [128, 256]   # 告诉 Patchcore 输出通道数

    def forward(self, x):
        # 返回列表，兼容 Patchcore 的 feature_extractor 期望
        v0 = self.stem(x)
        v1 = self.layer1(v0)
        v2 = self.layer2(v1)
        t = self.tex(x)
        e = self.edge(x)
        t = F.interpolate(t, size=v2.shape[2:], mode='bilinear')
        e = F.interpolate(e, size=v2.shape[2:], mode='bilinear')
        t = self.tex_proj(t)
        e = self.edge_proj(e)
        fused = self.gate(v2, t, e)
        v3 = self.layer3(v2)
        return [fused, v3]

transform = ResizeOnly((128,128))
dm = MVTecAD(root="/home/lili/datasets/mvtec_ad", category="bottle",
              train_batch_size=4, eval_batch_size=4,
              train_augmentations=transform, val_augmentations=transform, test_augmentations=transform)

# 直接传入 backbone，让 Patchcore 使用自定义 backbone
model = Patchcore(backbone=MultiModalBackbone())

engine = Engine()
engine.train(model=model, datamodule=dm)
results = engine.test(model=model, datamodule=dm)
print("\n========== AMCD-Net 最终结果 ==========")
for k, v in results[0].items():
    print(f"{k}: {v}")
