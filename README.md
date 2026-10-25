深度学习基于YOLOv8的无人机垃圾检测系统
@[toc]
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/14de15384fe345208107e1f09ae7c0d9.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/93be9911f7b4431b8d9a405b041875ac.png)

**数据集描述：
 24000 张无人机垃圾检测数据集，包括 ['waste']，1 类** TXT格式
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/af0d660dcaef4f01bfd2f3466df0d201.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4a2858bd6b9a4e938dd7da61cb53cf46.png)

### **基于YOLOv8的无人机垃圾检测系统**

| **类别** | **内容** |
|----------|----------|
| **项目名称** | 基于YOLOv8的无人机垃圾检测系统 |
| **数据集规模** | 24,000 张图像 |
| **检测类别** | `waste`（垃圾），共 1 类 |
| **数据集类型** | 无人机航拍图像（真实场景） |
| **标注格式** | YOLO 格式（`.txt`） |
| **模型状态** | 已训练完成，模型权重已生成，可直接调用 |
| **适用任务** | 垃圾目标检测、环境监测、无人机智能巡检 |

**---
**预实现功能：****
### **开发环境与软件依赖**

| 项目 | 配置要求 |
|------|----------|
| **开发软件** | PyCharm + Anaconda 或 VSCode + Anaconda |
| **Python 版本** | Python 3.9 |
| **核心库** | `ultralytics`（YOLOv8）、`torch==1.9`、`opencv-python`、`PyQt5` |
| **GUI 框架** | PyQt5（用于构建图形用户界面） |
| **运行平台** | Windows / Linux / macOS（支持主流操作系统） |

---

### **项目文件结构**

| 文件类型 | 包含内容 |
|----------|----------|
| **程序文件** | `.py` 源代码文件（主程序、检测逻辑、UI逻辑等） |
| **UI界面文件** | `.ui`（Qt Designer设计文件）、`.qrc`（资源文件）、转换后的 `.py` 界面代码 |
| **数据集** | 24,000 张带标签的无人机垃圾图像（图片 + `.txt` 标签） |
| **模型文件** | 训练好的 YOLOv8 模型权重（如 `best.pt`） |
| **辅助文件** | 项目运行教程（`.txt`）、效果图（`.jpg`）、图标资源等 |

---

### **预实现系统功能列表**

| 功能编号 | 功能描述 |
|----------|----------|
| ① | 支持单张图片或整个图片文件夹批量检测 |
| ② | 支持视频文件输入并进行逐帧检测 |
| ③ | 支持调用本地摄像头实时检测（视频流输入） |
| ④ | 支持自定义设置**置信度阈值**（Confidence）和 **IOU阈值**（用于NMS） |
| ⑤ | 支持选择是否显示**类别标签**和**检测框**，以及是否显示原图 |
| ⑥ | 支持加载和切换不同检测模型（如不同版本的 `.pt` 文件） |
| ⑦ | 批量检测后可在界面中**逐张查看每张图片的检测结果**（可视化展示） |
| ⑧ | 支持将检测结果（图像、视频、标注信息）**保存到本地** |

---

### **界面显示信息**

| 显示内容 | 说明 |
|----------|------|
| 目标位置 | 在图像上用边界框标出垃圾位置 |
| 目标总数 | 实时显示当前画面中检测到的垃圾总数 |
| 置信度 | 显示每个检测框的置信度分数（0~1） |
| 实时反馈 | 视频或摄像头模式下实现近实时检测与显示 |


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4b9bc5bcb2954da19bf883081692df69.png)
基于YOLOv8的无人机垃圾检测系统

## 🚀 二、环境搭建与运行步骤

### 1. **安装 Anaconda**

确保已安装 Anaconda，并创建一个新环境：

```bash
conda create -n drone_waste python=3.9
conda activate drone_waste
```

### 2. **安装依赖库**

在激活的环境中安装所有必要的库：

```bash
pip install -r requirements.txt
```

### 3. **运行主程序**

打开 PyCharm 或 VSCode，加载项目并运行 `MainProgram.py`：

```python
# MainProgram.py
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
```

### 4. **使用 UI 界面**

- **选择单张图片或图片文件夹**：点击相应按钮，选择要检测的图片或文件夹。
- **选择视频文件**：选择本地视频文件进行检测。
- **调用本地摄像头**：实时检测摄像头画面中的垃圾。
- **自定义置信度和 IOU 阈值**：调整检测精度。
- **显示标签和原图**：切换显示模式。
- **选择检测模型**：从下拉菜单中选择预训练模型。
- **查看批量检测结果**：检测完成后，可以在界面上查看每一张图片的结果。

---

## 📊 三、功能详解

### 1. **图片检测 (`imgTest.py`)**

```python
# imgTest.py
from ultralytics import YOLO
import cv2

def detect_image(image_path, model_path):
    model = YOLO(model_path)
    results = model(image_path)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"waste: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 2. **视频检测 (`VideoTest.py`)**

```python
# VideoTest.py
from ultralytics import YOLO
import cv2

def detect_video(video_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"waste: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Detection Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

### 3. **摄像头检测 (`CameraTest.py`)**

```python
# CameraTest.py
from ultralytics import YOLO
import cv2

def detect_camera(model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"waste: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Detection Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```
好的，以下是针对 **24000 张无人机拍摄的垃圾检测数据集**（YOLO 格式，1 类 `waste`）的完整训练方案，包括详细的训练代码、环境配置建议、数据组织结构、参数调优策略，以及后续推理和评估方法。

---

## ✅ 一、数据集结构（YOLO 格式）

请确保你的数据集按如下结构组织：

```
datasets/
└── drone_waste/
    ├── images/
    │   ├── train/      # 19200 张（80%）
    │   └── val/        # 4800 张（20%）
    ├── labels/
    │   ├── train/      # 对应的 .txt 标签文件
    │   └── val/
    └── data.yaml       # 数据集配置文件
```

> 📌 建议：训练集和验证集比例为 8:2，即 19200 / 4800。

---

### ✅ `data.yaml` 文件内容

```yaml
# datasets/drone_waste/data.yaml
train: ../datasets/drone_waste/images/train
val: ../datasets/drone_waste/images/val

nc: 1
names: ['waste']
```

---

## 🚀 二、环境准备（推荐使用 Anaconda）

```bash
# 1. 创建虚拟环境
conda create -n drone-waste-detect python=3.9
conda activate drone-waste-detect

# 2. 安装 PyTorch（支持 CUDA）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 Ultralytics YOLOv8
pip install ultralytics

# 4. 安装其他依赖
pip install opencv-python matplotlib pillow tqdm tensorboard
```

✅ 验证 GPU 是否可用：
```python
import torch
print(torch.cuda.is_available())  # 应输出 True
```

---

## 🧠 三、YOLOv8 训练代码（`train.py`）

由于是 **无人机视角下的小目标垃圾检测**，建议使用 **高分辨率输入 + 中大型模型** 来提升小目标检测能力。

```python
# train.py
from ultralytics import YOLO
import torch

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 加载预训练模型（推荐使用 yolov8m 或 yolov8l 提升小目标性能）
model = YOLO('yolov8m.pt')  # 可替换为 yolov8l.pt 或 yolov8x.pt

# 开始训练
results = model.train(
    data='datasets/drone_waste/data.yaml',   # 指向你的 data.yaml
    epochs=300,                              # 大数据集可训练更久
    imgsz=640,                               # 可尝试 1280（需大显存）
    batch=32,                                # 显存不足可降为 16 或 8
    name='drone_waste_yolov8m_640',          # 实验名称
    device=device,
    workers=8,
    optimizer='AdamW',                       # 更稳定收敛
    lr0=0.001,                               # 初始学习率
    weight_decay=0.0005,
    momentum=0.937,
    patience=50,                             # 早停：50 轮无提升则停止
    project='runs/drone_waste',
    amp=True,                                # 混合精度训练（加速）
    close_mosaic=30,                         # 最后30轮关闭 Mosaic 增强
    # --- 数据增强 ---
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.5,
    translate=0.2,
    scale=0.9,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,                               # 小目标增强
    degrees=10.0,
    # --- 缓存 ---
    cache='disk',                            # 缓存图像到磁盘，加快训练
)
```

---

## 🔍 四、训练策略优化（针对无人机小目标）

| 优化项 | 推荐设置 | 说明 |
|-------|---------|------|
| 输入尺寸 | `imgsz=640` 或 `1280` | 更高分辨率利于小目标检测 |
| 模型选择 | `yolov8m` / `yolov8l` | 更强特征提取能力 |
| Batch Size | `32` (640) / `16` (1280) | 根据显存调整 |
| 数据增强 | `mosaic=1.0`, `mixup=0.2`, `scale=0.9` | 提升泛化能力 |
| 学习率 | `lr0=0.001` | AdamW 适合小数据集 |
| 早停 | `patience=50` | 防止过拟合 |
| 混合精度 | `amp=True` | 加速训练，节省显存 |

> 💡 若显存不足，可降低 `imgsz` 或 `batch`，或使用 `yolov8s.pt`。

---

## 📊 五、模型评估（`val.py`）

```python
# val.py
from ultralytics import YOLO

model = YOLO('runs/drone_waste/drone_waste_yolov8m_640/weights/best.pt')

# 在验证集上评估
metrics = model.val(
    data='datasets/drone_waste/data.yaml',
    split='val',
    imgsz=640,
    batch=32,
    conf=0.001,
    iou=0.6,
    device=device,
    project='runs/eval',
    name='drone_waste_val'
)

print(f"✅ mAP@0.5:      {metrics.box.map:.4f}")
print(f"✅ mAP@0.5:0.95: {metrics.box.map50_95:.4f}")
print(f"✅ Precision:    {metrics.box.p:.4f}")
print(f"✅ Recall:       {metrics.box.r:.4f}")
```

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c0f0813b42b14871b4bb5986a5dba3a7.png)

## 🔎 六、推理代码（`predict.py`）

### 1. 单图检测

```python
# predict.py
from ultralytics import YOLO

model = YOLO('runs/drone_waste/drone_waste_yolov8m_640/weights/best.pt')

results = model.predict(
    source='test.jpg',
    save=True,
    conf=0.3,
    iou=0.45,
    show=False,
    project='runs/predict',
    name='waste_test'
)

for result in results:
    boxes = result.boxes
    print(f"检测到 {len(boxes)} 个垃圾")
    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy.cpu().numpy()[0]
        print(f"  位置: [{xyxy}], 置信度: {conf:.3f}")
```

### 2. 视频检测（无人机视频）

```python
results = model.predict(
    source='drone_video.mp4',
    save=True,
    conf=0.3,
    project='runs/predict_video',
    name='drone_waste_detection'
)
```

### 3. 实时摄像头检测

```python
results = model.predict(
    source=0,  # 本地摄像头
    show=True,
    conf=0.3,
    stream=True  # 流式处理
)

for r in results:
    print(f"检测到 {len(r.boxes)} 个垃圾")
```

---

## 📦 七、模型导出（用于部署）

```python
# export.py
from ultralytics import YOLO

model = YOLO('runs/drone_waste/drone_waste_yolov8m_640/weights/best.pt')

# 导出为 ONNX 格式（通用部署）
model.export(
    format='onnx',
    imgsz=640,
    dynamic=True,
    simplify=True
)

# 可选：导出为 TensorRT（高性能）
# model.export(format='engine', device=0)
```

生成 `best.onnx`，可用于：
- OpenCV DNN
- ONNX Runtime
- 边缘设备（Jetson、RK3588）
- Web 端（WebAssembly + ONNX.js）

---


