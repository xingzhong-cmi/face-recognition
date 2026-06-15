# 基于 PyTorch + InceptionResnetV1 的实时人脸识别

一个简洁的人脸识别小项目：

- 用 **VGGFace2 预训练 InceptionResnetV1** 提取人脸特征 embedding
- 用 **余弦相似度** 在已注册人脸库中匹配身份
- 通过 **笔记本摄像头** 进行实时检测与识别

## 技术栈

| 模块 | 实现 |
|------|------|
| 人脸检测 | [MTCNN](https://github.com/timesler/facenet-pytorch) |
| 特征提取 | facenet-pytorch 的 **InceptionResnetV1(pretrained='vggface2')** |
| 身份匹配 | 余弦相似度 + 阈值 |
| 摄像头 & 显示 | OpenCV |

## 安装

```bash
pip install -r requirements.txt
```

> Windows / 国内用户如下载缓慢，可改用清华镜像：
>
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

## 使用步骤

### 1. 准备已知人员的照片

在 `data/known_faces/` 下，为每个人创建一个文件夹（文件夹名 = 姓名），并放入若干张清晰的正脸照片。建议 **每人 3~5 张**，尽量覆盖不同光照与角度：

```
data/known_faces/
├── alice/
│   ├── 1.jpg
│   └── 2.jpg
└── bob/
    └── 1.jpg
```

### 2. 注册（构建人脸库）

```bash
python register.py
```

首次运行会自动下载 InceptionResnetV1 / MTCNN 的预训练权重（需要联网）。完成后会生成 `data/face_db.pt`。

### 3. 打开摄像头实时识别

```bash
python recognize.py
```

可选参数：

| 参数 | 默认 | 含义 |
|------|------|------|
| `--camera` | `0` | 摄像头索引；外接 USB 摄像头通常是 `1` |
| `--threshold` | `0.7` | 余弦相似度阈值，越大越严格；低于阈值显示 `Unknown` |
| `--detect-every` | `1` | 每 N 帧检测一次；CPU 卡顿时可调大（例如 `2` 或 `3`） |

按窗口中的 **`q` 键** 即可退出。

## 项目结构

```
face-recognition/
├── requirements.txt
├── README.md
├── .gitignore
├── model.py            # 人脸特征提取模型封装（InceptionResnetV1）
├── face_database.py    # 人脸库管理 (FaceDatabase)
├── register.py         # 注册新面孔（生成 data/face_db.pt）
├── recognize.py        # 实时摄像头识别主程序
└── data/
    └── known_faces/    # 已注册人员图片目录 (用户自行填充)
```

## 设计要点

1. **为什么不用分类头？**
   人脸识别是一个 *开集* 问题——新来的人不在训练集里。
   用 embedding + 相似度比对，新增人员只需添加照片重新注册，
   **无需重新训练模型**。

2. **MTCNN 与 InceptionResnetV1 的分工**
   - MTCNN 负责 *检测*："画面里哪里有人脸"
   - InceptionResnetV1 负责 *识别*："这张脸是谁"
   两者解耦，便于单独替换或升级。

3. **设备自适应**
   `model.get_device()` 会自动选择 CUDA → Apple Silicon MPS → CPU。

4. **性能优化**
   - `--detect-every N` 可以跳帧检测，CPU 上能从 ~5 FPS 提升到 ~20 FPS。
   - 跳过的帧仍会复用上一帧的检测结果绘制，画面不会闪烁。

## 进一步提升精度

- 每人尽量注册 3~5 张清晰正脸，覆盖光照与角度变化。
- 识别误报较多时，可把 `--threshold` 提高到 `0.75` 或 `0.8`。
- 若有更高实时性需求，可增大 `--detect-every`，在速度与稳定性之间折中。
- 如果有私有高质量人脸数据集，可在现有 embedding 基础上继续做度量学习微调。

## 常见问题

- **打不开摄像头？** 尝试 `--camera 1`；macOS 第一次运行需在「系统设置 → 隐私 → 摄像头」中授权终端。
- **升级后识别异常/全部 Unknown？** 若你升级前已经注册过，请重新运行 `python register.py`。新版本使用 VGGFace2 预训练模型，embedding 从 128 维改为 512 维，旧 `data/face_db.pt` 不兼容（程序会自动检测并提示）。
- **识别全部是 Unknown？** 多注册几张照片，或适当把 `--threshold` 调低（例如 `0.65`）。
- **总是把陌生人识别成已知人员？** 把 `--threshold` 调大（例如 `0.75`）。
- **没有 GPU 也能跑吗？** 可以。CPU 上建议配合 `--detect-every 2` 以保证流畅度。
