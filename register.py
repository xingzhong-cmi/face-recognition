"""
register.py
===========
注册脚本 —— 扫描 data/known_faces 目录，为每位已知人员构建特征库。

使用方法
--------
1. 在 ``data/known_faces/`` 下，为每个人创建一个文件夹
   （文件夹名就是这个人的姓名 / 标签）。
2. 把这个人若干张清晰的正脸照片放进去（建议 3~5 张，覆盖不同光照/角度）。
3. 在终端运行：

       python register.py

   完成后会生成 ``data/face_db.pt``，供 ``recognize.py`` 加载使用。

注意
----
首次运行会自动下载 InceptionResnetV1 与 MTCNN 的预训练权重，需要联网。
"""

from pathlib import Path

import torch
from pickle import UnpicklingError

from face_database import FaceDatabase
from model import FACE_EMBEDDING_SIZE


def _is_legacy_face_db(db_path: Path) -> bool:
    """检测当前 face_db.pt 是否为旧版本格式或旧维度。"""
    if not db_path.exists():
        return False

    try:
        payload = torch.load(db_path, map_location="cpu")
    except (RuntimeError, UnpicklingError, EOFError, ValueError):
        # 文件损坏/不可读也按不兼容处理，后续重建会覆盖。
        return True

    if isinstance(payload, dict) and "embeddings" in payload:
        return (
            int(payload.get("version", 0)) < FaceDatabase.DB_VERSION
            or int(payload.get("embedding_size", 0)) != FACE_EMBEDDING_SIZE
        )

    # 旧格式：{name: Tensor(N, 128)}
    return True


def main() -> None:
    print("=" * 60)
    print("开始构建人脸库 ...")
    print("=" * 60)

    db_path = Path("data/face_db.pt")
    if _is_legacy_face_db(db_path):
        print("[INFO] 检测到旧版本人脸库，将重新构建。")

    db = FaceDatabase()
    db.build()

    print("=" * 60)
    print(f"✓ 注册完成！共注册 {len(db.embeddings)} 人：")
    for name, embs in db.embeddings.items():
        print(f"    - {name}  ({embs.shape[0]} 张照片)")
    print("=" * 60)
    print("下一步：运行 `python recognize.py` 即可打开摄像头实时识别。")


if __name__ == "__main__":
    main()
