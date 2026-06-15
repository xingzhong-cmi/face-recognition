"""
recognize.py
============
实时人脸识别主程序。

流程
----
  1. 用 OpenCV 打开笔记本摄像头
  2. 每一帧（或每 N 帧）用 MTCNN 检测所有人脸
  3. 对每张人脸用 ResNet 提取 embedding，并在人脸库中找最相似的人
  4. 用 OpenCV 绘制人脸框 + 姓名 + 相似度
  5. 按 'q' 退出

使用方法
--------
    python recognize.py                          # 默认摄像头 0，阈值 0.6
    python recognize.py --camera 1               # 使用外接摄像头
    python recognize.py --threshold 0.7          # 收紧识别阈值
    python recognize.py --detect-every 2         # 每 2 帧检测一次 (省 CPU)
"""

import argparse
import time

import cv2
import torch  # noqa: F401  # 让 torch 提前初始化，避免首帧延迟过大
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image

from face_database import FaceDatabase
from model import face_transform, get_device


# ----------------------------------------------------------------------
# 工具函数：在帧上绘制人脸框与标签
# ----------------------------------------------------------------------
def draw_box(frame, box, label: str, color=(0, 255, 0)) -> None:
    """在 BGR 图像上画一个矩形框 + 顶部标签条。

    Parameters
    ----------
    frame : np.ndarray
        OpenCV 的 BGR 图像（会被原地修改）。
    box : list/tuple[float]
        [x1, y1, x2, y2] 形式的 bbox 坐标（可能是浮点）。
    label : str
        要显示的文字，例如 "alice (0.83)"。
    color : tuple[int, int, int]
        BGR 颜色；已知人脸用绿色，陌生人用红色。
    """
    x1, y1, x2, y2 = [int(v) for v in box]

    # 1) 画框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 2) 标签背景条，方便阅读
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main() -> None:
    # ----- 命令行参数 -----
    parser = argparse.ArgumentParser(description="基于 PyTorch+ResNet 的实时人脸识别")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头索引；笔记本内置一般为 0，外接 USB 摄像头通常为 1")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="余弦相似度阈值，低于该值则判为 Unknown；越大越严格")
    parser.add_argument("--detect-every", type=int, default=1,
                        help="每 N 帧执行一次检测+识别；CPU 卡顿时可调大")
    args = parser.parse_args()

    # ----- 设备 -----
    device = get_device()
    print(f"[INFO] 使用设备: {device}")

    # ----- 加载人脸库 -----
    db = FaceDatabase()
    if not db.load():
        print("[ERROR] 未找到人脸库 (data/face_db.pt)，请先运行: python register.py")
        return
    if not db.embeddings:
        print("[ERROR] 人脸库为空，请先在 data/known_faces/ 下添加图片并运行 register.py")
        return

    # ----- 多人脸检测器 -----
    # 这里 keep_all=True，表示一帧里可能有多张脸，全部返回。
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)

    # ----- 打开摄像头 -----
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 (index={args.camera})")
        return

    print("[INFO] 摄像头已开启，按 'q' 退出 ...")

    frame_count = 0
    # 缓存上一次检测结果，省去每帧都跑模型
    last_results = []  # list of (box, name, score)

    # 简易 FPS 计算
    fps_t0 = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取摄像头帧失败，退出。")
            break
        frame_count += 1

        # ----- 每 N 帧检测一次 -----
        if frame_count % args.detect_every == 0:
            # OpenCV 是 BGR，PIL/torch 默认 RGB，需要转一下
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            # mtcnn.detect 返回所有人脸的 bbox 和置信度；可能为 None
            boxes, _probs = mtcnn.detect(pil)
            # mtcnn(pil) 返回裁剪好的人脸张量 (N, 3, 160, 160) 或 None
            faces = mtcnn(pil)

            results = []
            if boxes is not None and faces is not None:
                for box, face in zip(boxes, faces):
                    # MTCNN 输出范围约 [-1, 1]，反归一化回 [0, 1] -> PIL
                    # 再走我们自己的 face_transform，确保和注册阶段一致
                    face_img = (face.clamp(-1, 1) + 1) / 2
                    pil_face = to_pil_image(face_img)
                    tensor = face_transform(pil_face)
                    name, score = db.identify(tensor, threshold=args.threshold)
                    results.append((box, name, score))
            last_results = results

        # ----- 在原始帧上绘制（包括「跳帧」的中间帧，避免画面闪烁） -----
        for box, name, score in last_results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({score:.2f})"
            draw_box(frame, box, label, color)

        # ----- 计算并显示 FPS -----
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10.0 / max(now - fps_t0, 1e-6)
            fps_t0 = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # ----- 显示窗口 -----
        cv2.imshow("Face Recognition (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ----- 收尾 -----
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 已退出。")


if __name__ == "__main__":
    main()
