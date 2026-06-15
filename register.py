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
首次运行会自动下载 ResNet 与 MTCNN 的预训练权重，需要联网。
"""

from face_database import FaceDatabase


def main() -> None:
    print("=" * 60)
    print("开始构建人脸库 ...")
    print("=" * 60)

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
