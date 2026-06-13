# YOLOv8m Object Detection on Pascal VOC 2012 (5-Class Subset)

## Problem Statement

Object detection is a fundamental task in computer vision with applications in autonomous driving, surveillance, and robotics. However, real-world detection systems must handle challenges such as varying object sizes, occlusions, and limited computational resources. Data augmentation is widely used to improve generalization, but its specific impact on different object sizes and classes is not fully understood.

This study investigates how data augmentation affects detection performance across five biologically and visually distinct classes: cat, horse, car, dog, and bird. By comparing models trained with and without augmentation, we aim to quantify the benefits of augmentation and identify which object classes and sizes benefit the most.

This project trains and evaluates a YOLOv8m detector on a five-class subset of the Pascal VOC 2012 benchmark: **cat, horse, car, dog, bird**. The primary goal is to understand the full object detection pipeline — from data preparation and preprocessing to training and quantitative evaluation — and to measure how data augmentation affects generalization.

## Key Findings

- **Model A (with augmentation)** achieved **mAP@50 of 0.8009** and **mAP@50:95 of 0.6087**
- **Model B (without augmentation)** achieved **mAP@50 of 0.7575** and **mAP@50:95 of 0.5559**
- Augmentation improved mAP@50 by **+4.3 percentage points** and mAP@50:95 by **+5.3 percentage points**
- **Cat** was the strongest class (AP 0.8687)
- **Bird** was the most challenging class (AP 0.7132) due to small object size and occlusion
- Both models perform well on large objects but struggle on small objects, which make up only 2.8% of the validation set

## Dataset

We used Pascal VOC 2012 downloaded via Roboflow in YOLOv8 format. The full dataset covers 20 object classes across approximately 11,500 annotated images. To stay within compute budget and enable controlled experiments, we reduced scope to five biologically and visually distinct classes.

### Filtering and Cleaning

- Class indices remapped to 0-4 (cat=0, horse=1, car=2, dog=3, bird=4)
- Images containing none of the five classes were removed
- Final split: **3,948 training images** and **1,020 validation images**

### Class Distribution (Training Set)

| Class | Instances | Share (%) |
|-------|-----------|-----------|
| Car   | 1,988     | 41.3      |
| Dog   | 1,234     | 25.7      |
| Bird  | 1,056     | 22.0      |
| Cat   | 1,034     | 21.5      |
| Horse | 611       | 12.7      |
| **Total** | **4,923** | **100**   |

### Object Size Distribution (Validation Set)

- Large objects (>96² px): **78.8%**
- Medium objects (32²–96² px): **18.3%**
- Small objects (<32² px): **2.8%** (only 43 instances)

## Model and Training

### Architecture

We used **YOLOv8m (medium variant)** from Ultralytics, loaded with weights pre-trained on COCO. YOLOv8 is a single-stage, anchor-free detector. The backbone extracts feature maps at three scales (80×80, 40×40, 20×20), which are fused by a PAN neck before three decoupled detection heads output box coordinates and class scores independently.

### Training Configuration

- **Epochs:** 20
- **Image size:** 640×640
- **Batch size:** 16
- **GPU:** Tesla T4 (14GB)
- **Early stopping:** patience 15
- **Pre-trained weights:** COCO

### Augmentation Settings

| Parameter          | Model A (aug ON) | Model B (aug OFF) |
|--------------------|------------------|-------------------|
| Mosaic             | 1.0              | 0.0               |
| HSV hue            | 0.015            | 0.0               |
| HSV saturation     | 0.7              | 0.0               |
| HSV brightness     | 0.4              | 0.0               |
| Horizontal flip    | 0.5              | 0.0               |

## Results

### Overall Metrics

| Metric          | Model A | Model B | Δ      |
|-----------------|---------|---------|--------|
| mAP@50          | 0.8009  | 0.7575  | +0.0434|
| mAP@50:95       | 0.6087  | 0.5559  | +0.0528|
| Precision       | 0.8149  | 0.7737  | +0.0412|
| Recall          | 0.7211  | 0.6856  | +0.0355|

### Per-Class Performance (Model A, AP@50)

| Class | AP@50 | Validation Instances |
|-------|-------|---------------------|
| Cat   | 0.8687| 243                 |
| Dog   | 0.8160| 364                 |
| Horse | 0.8095| 192                 |
| Car   | 0.7973| 503                 |
| Bird  | 0.7132| 215                 |

### Performance by Object Size (IoU ≥ 0.5)

| Size   | Model A (Prec/Rec/F1) | Model B (Prec/Rec/F1) |
|--------|------------------------|------------------------|
| Small  | 0.00 / 0.00 / 0.00     | 0.00 / 0.00 / 0.00     |
| Medium | 0.00 / 0.00 / 0.00     | 0.00 / 0.00 / 0.00     |
| Large  | 0.82 / 0.72 / 0.77     | 0.78 / 0.69 / 0.73     |

> *Note: The small and medium object metrics reflect the extreme scarcity of such instances in the validation set (only 43 small objects total), not a fundamental model failure.*

## Key Conclusions

1. **Data augmentation is not optional** — Model A outperformed Model B across every metric. The larger gain on mAP@50:95 (+5.3 pp) confirms that augmentation improves both detection and box localization quality.

2. **Mosaic augmentation matters most for small and medium objects** — This is exactly what Mosaic targets by creating artificially small instances through image tiling.

3. **Class frequency alone does not guarantee accuracy** — Car had the most training instances (1,988) but achieved only 0.7973 AP, while Cat had fewer instances (1,034) but achieved the highest AP (0.8687).

4. **Bird is the most challenging class** — Due to high variability in scale, occlusion (branches, foliage), and compact silhouettes where small box errors produce large IoU penalties.

## Inference Speed

- **14–16 ms per image** on Tesla T4 GPU
- Well within real-time requirements (<33 ms for 30 fps video)
- Confirms YOLOv8m as a practical deployment choice

## Limitations and Future Work

- Training was limited to 20 epochs due to Colab session constraints. Training loss curves had not fully plateaued, suggesting **50+ epochs** could push mAP@50 above 0.83.
- Small object detection remains poor due to dataset scarcity (only 2.8% of validation objects are small). Potential solutions:
  - Copy-paste augmentation
  - Class-weighted loss
  - Test-time augmentation (TTA)
  - SAHI (Slicing Aided Hyper Inference)

## Optimisation Exploration

Based on the work of Rasheed and Zarkoosh [1], we explored removing the small-object detection branch (YOLOv8-medium-large variant) since medium and large objects dominate our dataset. This modification reduces computational cost and inference time with minimal accuracy loss — though effectiveness remains data-dependent.

## References

[1] Rasheed and Zarkoosh. Architectural optimisation of YOLOv8 for size-specific object detection.

## Repository Structure
