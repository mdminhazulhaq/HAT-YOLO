# HAT-YOLO
HAT-YOLO is an improved lightweight YOLOv8n model that integrates dual channel–spatial attention, lightweight transformer-based attention, and GELU-enhanced modules for efficient UAV object detection on low-memory devices. Trained on VEDAI and RSOD datasets, it achieves higher mAP accuracy while running at real-time detection speed. 

**Environmental Setup tools**
This setup is designed for a Windows 10 environment with an NVIDIA RTX A2000 GPU.

1. Install Anaconda3 (Python 3.10 recommended)
2. Install Git-2.26.2-64-bit
3. Install NVIDIA GPU Driver
4. Install CUDA 12.8
5. Install cuDNN 8.9.7 (compatible with CUDA 12.8)
6. Create a new Conda environment (Python 3.10). We created an environment named yolov8.
7. Open the Anaconda Prompt, activate the environment:

```
   activate yolov8
```
8. Navigate to the directory where your YOLO project will be stored. Example: E directory

```
   E:
```
9. Create the root folder (optional but recommended):

```
   mkdir yolov8-gpu
```

10. Install CUDA-supported PyTorch (example with cu118):

```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

11. Install Ultralytics:

```
   pip install ultralytics
```

**Dataset**
The VEDAI and RSOD datasets are large, so they are not uploaded into the repository.
You may download them from publicly available sources and label the images into YOLO format.

**Data Preparation**
1. Place the images and labels into the following structure:

```
   dataset/
│── VEDAI/
│   ├── images/
│   │    ├── train/
│   │    ├── val/
│   │    └── test/
│   └── labels/
│        ├── train/
│        ├── val/
│        └── test/
│
│── RSOD/
    ├── images/
    │    ├── train/
    │    ├── val/
    │    └── test/
    └── labels/
         ├── train/
         ├── val/
         └── test/
```

2. Create dataset .yaml files for YOLO.
  Example VEDAI YAML file

```
   # VEDAI Dataset Path
path: dataset/VEDAI-DATASET

train: images/train
val: images/val
test: images/test

names: ['car', 'truck', 'pickup', 'tractor', 'boat', 'van', 'other', 'campingCar']
```
Example RSOD YAML file
```
   # RSOD Dataset Path
path: dataset/RSOD-DATASET

train: images/train
val: images/val
test: images/test

names: ['aircraft', 'oil-tank', 'overpass', 'playground']
```

**Train and test of the HAT-YOLO model and YOLOv8n model using VEDAI dataset on NVIDIA RTX A2000 Laptop GPU**
1. Training command for the proposed HAT-YOLO model
```
yolo detect train model=HAT-YOLO.yaml data=dataset\VEDAI-DATASET.yaml imgsz=640 workers=8 batch=8 device=0 epochs=300 line_thickness=2 patience=300
```
2. Test command for HAT-YOLO model
```
yolo detect predict model=HAT-YOLO-best.pt source="E:\yolov8-gpu\dataset\VEDAI-DATASET\images\test" save=True
```
3. Training command for YOLOv8n model
```
yolo detect train model=yolov8n.yaml data=dataset\VEDAI-DATASET.yaml imgsz=640 workers=8 batch=8 device=0 epochs=300 line_thickness=2 patience=300
```
4. Test command for YOLOv8n model
```
yolo detect predict model=YOLOv8n-best.pt source="E:\yolov8-gpu\dataset\VEDAI-DATASET\images\test" save=True
```


**Train and test of the HAT-YOLO model and YOLOv8n model using RSOD dataset on NVIDIA RTX A2000 Laptop GPU**
1. Training command for the proposed HAT-YOLO model
```
yolo detect train model=HAT-YOLO.yaml data=dataset\RSOD-DATASET.yaml imgsz=640 workers=8 batch=8 device=0 epochs=300 line_thickness=2 patience=300
```
2. Test command for HAT-YOLO model
```
yolo detect predict model=HAT-YOLO-best.pt source="E:\yolov8-gpu\dataset\RSOD-DATASET\images\test" save=True
```
3. Training command for YOLOv8n model
```
yolo detect train model=yolov8n.yaml data=dataset\RSOD-DATASET.yaml imgsz=640 workers=8 batch=8 device=0 epochs=300 line_thickness=2 patience=300
```
4. Test command for YOLOv8n model
```
yolo detect predict model=YOLOv8n-best.pt source="E:\yolov8-gpu\dataset\RSOD-DATASET\images\test" save=True
```






