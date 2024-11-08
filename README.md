# Object Detection and Segmentation Project

This project focuses on training and comparing two different architectures for object detection: YOLOv8 and Faster R-CNN. We use these models for recognizing road signs and speed limits.

## Technologies

- PyTorch
- Ultralytics YOLOv8
- Detectron2
- OpenCV




##Usage
Launch Jupyter Notebook to start training and testing the models:**(ipynb_files)**
jupyter notebook yolov8_for_sign_detection.ipynb
jupyter notebook faster_rcnn_for_sign_detection.ipynb



##Data
The data for training and testing the models is located in the **dataset** folder. 



## Metrics
In this project, we use the following metrics to evaluate the performance of the models:

mAP (mean Average Precision)
Precision
Recall



## Results
You can see real-time result by launching Jupyter notebook in folder **short_deployment_with_webcam_for_yolov8**:
jupyter notebook  webcam_test_signs.ipynb

Also see result in the folder **screenshots**.

As a result of training the models, the following results were obtained:

YOLOv8
Mean Average Precision (mAP): 0.964
Precision for traffic signs: 0.831 for green light and 0.835 for red light.
Faster R-CNN
Mean Average Precision (mAP): 11.21
Precision for traffic signs: 11.80 for green light and 7.44 for red light.

## MODELS
2 models saved in folder **ready_to_use_models **
For Faster R-CNN: **faster_rcnn_resnet50_final.pth**
For Yolov8: **yolov8.onnx**

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
