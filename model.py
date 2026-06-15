"""
model.py
=========
人脸特征提取模型封装。

这里使用 facenet-pytorch 提供的 InceptionResnetV1(vggface2)：
- 模型在 VGGFace2 上训练，专门用于人脸识别任务
- 输出 512 维 embedding，且已做 L2 归一化
- 输入直接使用 MTCNN 的输出（范围约 [-1, 1]），无需再做 ImageNet 标准化
"""

import torch
from facenet_pytorch import InceptionResnetV1


FACE_EMBEDDING_SIZE = 512


def get_face_encoder(device: torch.device) -> InceptionResnetV1:
    """创建并返回人脸特征提取器。

    Parameters
    ----------
    device : torch.device
        模型运行设备。

    Returns
    -------
    InceptionResnetV1
        预训练人脸编码器（VGGFace2），已切到 eval 模式。
    """
    model = InceptionResnetV1(pretrained="vggface2").to(device)
    model.eval()
    return model


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
