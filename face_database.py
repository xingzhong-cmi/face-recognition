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
生成一个二进制文件，默认格式为：

    {
        "version": 2,
        "embedding_size": 512,
        "embeddings": {
            "alice": Tensor(N1, D),
            "bob":   Tensor(N2, D),
            ...
        }
    }
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN

from model import FACE_EMBEDDING_SIZE, get_device, get_face_encoder


class FaceDatabase:
    """人脸特征库的封装。

    Parameters
    ----------
    data_dir : str
        已注册人员图片所在的根目录。
    db_path : str
        生成的人脸库 (.pt) 文件保存路径。
    """
    DB_VERSION = 2

    def __init__(self,
                 data_dir: str = "data/known_faces",
                 db_path: str = "data/face_db.pt"):
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)
        self.device = get_device()
        self.embedding_size = FACE_EMBEDDING_SIZE
        self.db_version = self.DB_VERSION

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
        self.model = get_face_encoder(self.device)

        # 已注册人脸库：name -> Tensor(N, D)
        self.embeddings: Dict[str, torch.Tensor] = {}

    # ==================================================================
    # 内部工具：从一张 PIL 图片提取人脸 embedding
    # ==================================================================
    @torch.no_grad()
    def _extract_embedding(self, pil_image: Image.Image) -> Optional[torch.Tensor]:
        """先用 MTCNN 检测并裁剪人脸，再提取 embedding。

        Returns
        -------
        torch.Tensor 或 None
            (D,) 形状的特征向量；若未检测到人脸则返回 None。
        """
        # MTCNN 输出形状 (3, 160, 160)，像素范围约为 [-1, 1]
        face = self.mtcnn(pil_image)
        if face is None:
            return None

        # InceptionResnetV1 直接吃 MTCNN 输出，无需二次标准化。
        tensor = face.unsqueeze(0).to(self.device)  # (1, 3, 160, 160)

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
        payload = {
            "version": self.DB_VERSION,
            "embedding_size": self.embedding_size,
            "embeddings": self.embeddings,
        }
        torch.save(payload, self.db_path)
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
            payload: Any = torch.load(self.db_path, map_location="cpu")
            if isinstance(payload, dict) and "embeddings" in payload:
                self.db_version = int(payload.get("version", 0))
                self.embedding_size = int(payload.get("embedding_size", 0))
                self.embeddings = payload.get("embeddings", {})
                if not isinstance(self.embeddings, dict):
                    raise ValueError(f"人脸库格式不支持: {self.db_path}")
            elif isinstance(payload, dict) and all(
                    isinstance(v, torch.Tensor) for v in payload.values()):
                # 兼容旧格式：{name: Tensor(N, 128)}
                self.db_version = 1
                self.embeddings = payload
                if self.embeddings:
                    first = next(iter(self.embeddings.values()))
                    self.embedding_size = int(first.shape[-1])
            else:
                raise ValueError(f"人脸库格式不支持: {self.db_path}")
            print(f"已加载人脸库: {self.db_path}，共 {len(self.embeddings)} 人")
            return True
        return False

    def is_compatible(self) -> bool:
        """当前加载的人脸库是否与新版编码器兼容。"""
        return (
            self.db_version >= self.DB_VERSION
            and self.embedding_size == FACE_EMBEDDING_SIZE
        )

    # ==================================================================
    # 识别
    # ==================================================================
    @torch.no_grad()
    def identify(self,
                 face_tensor: torch.Tensor,
                 threshold: float = 0.7) -> Tuple[str, float]:
        """对一张已经裁剪好的人脸，做身份识别。

        Parameters
        ----------
        face_tensor : torch.Tensor
            形状 (3, 160, 160) 的人脸张量（MTCNN 输出，范围约 [-1, 1]）。
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
        has_valid_match = False
        for name, embs in self.embeddings.items():
            if embs.ndim != 2 or embs.shape[1] != query.shape[0]:
                print(f"[WARN] 跳过维度不匹配的人脸数据: {name}")
                continue
            has_valid_match = True
            # embs: (N, D), query: (D,)
            # 由于 query / embs 都已经 L2 归一化，矩阵乘积即为余弦相似度
            scores = embs @ query                      # (N,)
            max_score = scores.max().item()            # 多照片最高分
            # 均值向量可降低单张异常照片影响，max_score 可保留最优匹配能力。
            mean_emb = F.normalize(embs.mean(dim=0), p=2, dim=0)
            mean_score = torch.dot(mean_emb, query).item()
            score = max(max_score, mean_score)
            if score > best_score:
                best_score = score
                best_name = name

        if not has_valid_match:
            return "Unknown", 0.0

        # 低于阈值认为是陌生人
        if best_score < threshold:
            return "Unknown", best_score
        return best_name, best_score
