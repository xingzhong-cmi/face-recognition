"""
model.py
=========
基于 ResNet 的人脸特征提取网络。

设计思路
--------
人脸识别是一个 **开集问题**（新来的人可能从未出现在训练集里），
因此我们不使用「N 类分类头」的做法，而是把网络当作一个
**特征提取器（embedding extractor）**：

    输入  : 一张已经裁剪好的人脸图  (3, H, W)
    输出  : 一个 D 维的特征向量    (D,)，并做 L2 归一化

识别阶段只需把摄像头里检测到的人脸 embedding 与人脸库中
各人 embedding 做 **余弦相似度** 比对，相似度最高且超过阈值
的那个人即为识别结果。

由于没有大规模人脸数据集做训练，这里直接复用 torchvision 在
ImageNet 上预训练好的 ResNet 作为 backbone，再接一层 Linear
投影到固定维度。对于「小规模、少量人」的场景，这种方案已经够用。
如需更高精度，可以：
  1) 换成在 VGGFace2/CASIA-WebFace 上预训练的人脸专用模型，或
  2) 用 Triplet Loss / ArcFace 在本网络上 fine-tune。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class FaceResNet(nn.Module):
    """基于 ResNet 的人脸特征提取网络。

    Parameters
    ----------
    embedding_size : int
        输出特征向量的维度，默认 128。维度越大区分能力越强，但越占内存。
    backbone : str
        "resnet18" 或 "resnet50"。前者轻量、CPU 也能跑；后者精度更高。
    pretrained : bool
        是否加载 ImageNet 预训练权重。强烈建议保持 True。
    """

    def __init__(self,
                 embedding_size: int = 128,
                 backbone: str = "resnet18",
                 pretrained: bool = True):
        super().__init__()

        # ---------- 1. 选择 backbone ----------
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features  # ResNet18 的 fc 输入维度 = 512
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features  # ResNet50 的 fc 输入维度 = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ---------- 2. 去掉原 ResNet 的分类头 ----------
        # base.children() 的最后一个元素是 fc (Linear)，倒数第二个是
        # AdaptiveAvgPool2d。我们只去掉最后的 fc，把 avgpool 保留下来，
        # 这样输出形状是 (B, in_features, 1, 1)。
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # ---------- 3. 接一个新的 embedding 层 ----------
        # 把 ResNet 的高维特征映射到 embedding_size 维。
        self.embedding = nn.Linear(in_features, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            形状 (B, 3, H, W) 的图像张量，已经过标准化预处理。

        Returns
        -------
        torch.Tensor
            形状 (B, embedding_size) 的特征向量，已做 L2 归一化，
            因此两两向量的点积就等于余弦相似度，便于后续比对。
        """
        x = self.backbone(x)            # (B, C, 1, 1)
        x = torch.flatten(x, 1)         # (B, C)
        x = self.embedding(x)           # (B, embedding_size)
        x = F.normalize(x, p=2, dim=1)  # L2 归一化 -> 单位向量
        return x


# ----------------------------------------------------------------------
# 标准的人脸预处理流水线
# ----------------------------------------------------------------------
# 注意：这里的均值/方差使用 ImageNet 的统计值，必须与 backbone 预训练时
# 的预处理保持一致，否则特征会失真。
face_transform = transforms.Compose([
    transforms.Resize((160, 160)),                # 统一尺寸
    transforms.ToTensor(),                        # PIL -> Tensor，且 [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_device() -> torch.device:
    """自动选择最优计算设备。

    优先级： CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # macOS M1/M2/M3 上的 GPU 加速
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
