"""
face_database.py
================
人脸库（FaceDatabase）：负责
  1. 加载 / 构建已注册人员的特征库
  2. 对一张未知人脸进行身份识别

目录约定
--------
    data/known_faces/
        alice/
            1.jpg
            2.jpg
        bob/
            1.jpg

每个一级子目录的名字即为该人员的姓名（label），目录下放该人的
若干张正脸照片即可。运行 ``register.py`` 后会在 ``data/face_db.pt``
生成一个二进制文件，里面是字典：

    {
        "alice": Tensor(N1, D),   # N1 张图，每张 D 维 embedding
        "bob":   Tensor(N2, D),
        ...
    }
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image

from model import FaceResNet, face_transform, get_device


class FaceDatabase:
    """人脸特征库的封装。

    Parameters
    ----------
    data_dir : str
        已注册人员图片所在的根目录。
    db_path : str
        生成的人脸库 (.pt) 文件保存路径。
    embedding_size : int
        embedding 的维度，需与 ``FaceResNet`` 保持一致。
    backbone : str
        "resnet18" 或 "resnet50"。
    """

    def __init__(self,
                 data_dir: str = "data/known_faces",
                 db_path: str = "data/face_db.pt",
                 embedding_size: int = 128,
                 backbone: str = "resnet18"):
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)
        self.device = get_device()

        # ---------- 人脸检测器 ----------
        # MTCNN 会从原图中找到人脸，并裁剪到 image_size 大小。
        # margin 是在 bbox 外多保留的像素，避免裁得太紧。
        # keep_all=False 表示构建数据库时每张图只取置信度最高的一张脸
        # （注册阶段我们假设每张图就是某个人的正脸特写）。
        self.mtcnn = MTCNN(image_size=160,
                           margin=20,
                           keep_all=False,
                           device=self.device)

        # ---------- 特征提取器 ----------
        self.model = FaceResNet(embedding_size=embedding_size,
                                backbone=backbone).to(self.device)
        # 没有训练阶段，直接 eval 即可（关闭 dropout/bn 更新）
        self.model.eval()

        # 已注册人脸库：name -> Tensor(N, D)
        self.embeddings: Dict[str, torch.Tensor] = {}

    # ==================================================================
    # 内部工具：从一张 PIL 图片提取人脸 embedding
    # ==================================================================
    @torch.no_grad()
    def _extract_embedding(self, pil_image: Image.Image) -> Optional[torch.Tensor]:
        """先用 MTCNN 检测并裁剪人脸，再用 ResNet 提取 embedding。

        Returns
        -------
        torch.Tensor 或 None
            (D,) 形状的特征向量；若未检测到人脸则返回 None。
        """
        # MTCNN 输出形状 (3, 160, 160)，像素范围约为 [-1, 1]
        face = self.mtcnn(pil_image)
        if face is None:
            return None

        # 反归一化到 [0, 1]，再走自己的 transform 走一次标准化，
        # 这样和实时识别阶段使用的预处理完全一致。
        face_img = (face.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        pil_face = to_pil_image(face_img)
        tensor = face_transform(pil_face).unsqueeze(0).to(self.device)

        emb = self.model(tensor).squeeze(0).cpu()  # (D,)
        return emb

    # ==================================================================
    # 构建 / 加载 / 保存
    # ==================================================================
    def build(self) -> None:
        """扫描 data_dir 目录，为每个人提取 embedding 并存到 self.embeddings。"""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"目录不存在: {self.data_dir}\n"
                f"请按照 data/known_faces/<姓名>/*.jpg 的结构放置照片。"
            )

        self.embeddings.clear()

        # 遍历每个人的子目录
        for person_dir in sorted(self.data_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            embs: List[torch.Tensor] = []

            for img_path in sorted(person_dir.iterdir()):
                # 只处理常见图片格式
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"  ✗ 跳过 {img_path.name}: {e}")
                    continue

                emb = self._extract_embedding(img)
                if emb is None:
                    print(f"  ✗ {img_path.name}: 未检测到人脸")
                    continue
                embs.append(emb)
                print(f"  ✓ {name}/{img_path.name}")

            if embs:
                # 把同一人的多张照片堆叠成 (N, D)
                self.embeddings[name] = torch.stack(embs)
                print(f"[{name}] 共 {len(embs)} 张有效图片")
            else:
                print(f"[{name}] 无有效图片，已跳过")

        self.save()

    def save(self) -> None:
        """把人脸库序列化到磁盘。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.embeddings, self.db_path)
        print(f"人脸库已保存到: {self.db_path}")

    def load(self) -> bool:
        """从磁盘加载人脸库。

        Returns
        -------
        bool
            True  : 成功加载
            False : 文件不存在
        """
        if self.db_path.exists():
            # weights_only=False 是因为我们存的是 dict，不是模型权重
            self.embeddings = torch.load(self.db_path, map_location="cpu")
            print(f"已加载人脸库: {self.db_path}，共 {len(self.embeddings)} 人")
            return True
        return False

    # ==================================================================
    # 识别
    # ==================================================================
    @torch.no_grad()
    def identify(self,
                 face_tensor: torch.Tensor,
                 threshold: float = 0.6) -> Tuple[str, float]:
        """对一张已经裁剪 + 标准化好的人脸，做身份识别。

        Parameters
        ----------
        face_tensor : torch.Tensor
            形状 (3, H, W)，由 face_transform 处理过的张量。
        threshold : float
            余弦相似度阈值；最佳匹配低于该值则返回 "Unknown"。
            该阈值需要根据实际场景微调，常见范围 0.5 ~ 0.8。

        Returns
        -------
        (name, score) : Tuple[str, float]
            name  : 识别到的姓名 (或 "Unknown")
            score : 与该姓名的最高余弦相似度 (在 [-1, 1] 之间，通常正数)
        """
        if not self.embeddings:
            return "Unknown", 0.0

        # 提取 query 特征
        tensor = face_tensor.unsqueeze(0).to(self.device)
        query = self.model(tensor).squeeze(0).cpu()  # (D,)

        # 在所有已注册人员中找余弦相似度最高的
        best_name = "Unknown"
        best_score = -1.0
        for name, embs in self.embeddings.items():
            # embs: (N, D), query: (D,)
            # 由于 query / embs 都已经 L2 归一化，矩阵乘积即为余弦相似度
            scores = embs @ query                # (N,)
            score = scores.max().item()          # 取该人多张照片中的最高分
            if score > best_score:
                best_score = score
                best_name = name

        # 低于阈值认为是陌生人
        if best_score < threshold:
            return "Unknown", best_score
        return best_name, best_score
