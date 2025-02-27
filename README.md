<<<<<<< HEAD
# VR_Assignment1_Akash-Upadhyay_MT2024013
=======
# Computer Vision Assignment: Coin Detection and Panorama Stitching

This repository contains two computer vision projects implemented in Python:
1. Coin Detection and Segmentation
2. Panorama Image Stitching

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Akash-Upadhyay/VR_Assignment1_Akash-Upadhyay_MT2024013.git
cd VR_Assignment1_Akash-Upadhyay_MT2024013
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

Required packages (automatically installed from requirements.txt):
- Python 3.x
- OpenCV (`cv2`) >= 4.5.0
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- imutils >= 0.5.4

### 4. Prepare Your Data
1. For Coin Detection:
   - Place your coin images in `coins_images/`

2. For Panorama Stitching:
   - Place your sequential images in `panorama_images/`

Required packages:
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- imutils

## 1. Coin Detection and Segmentation

### Features
- Automatic coin detection using edge detection
- Contour-based coin segmentation
- Individual coin extraction
- Visualization of detection results
- Coin counting functionality

### Implementation Details

#### Detection Process
1. Image Preprocessing:
   - Grayscale conversion
   - Gaussian blur (11x11 kernel)
   - Canny edge detection (thresholds: 30, 150)
   - Edge dilation

#### Segmentation Process
1. Contour Detection:
   - External contour finding
   - Area-based filtering (min_area: 1000 pixels)
2. Individual Coin Extraction:
   - Mask creation
   - Bounding box cropping

### Usage

```bash
python src/coin_detection.py
```

### Results
As shown in the output images:
- Successfully detected 4 Indian coins
- Clear segmentation with green contour highlighting
- Individual coin isolation in separate images
- Clean mask generation showing coin boundaries

## 2. Panorama Image Stitching

### Features
- SIFT-based feature detection and matching
- Homography computation using RANSAC
- Perspective warping for seamless stitching
- Support for multiple image stitching
- Keypoint match visualization

### Implementation Details

#### Stitching Process
1. Feature Detection:
   - SIFT keypoint detection
   - Feature descriptor computation
2. Feature Matching:
   - Brute force matcher with Lowe's ratio test
   - Homography computation using RANSAC
3. Image Warping:
   - Perspective transformation
   - Image blending

### Usage

```bash
python src/stitch.py
```
When prompted, enter the pattern for your image files (e.g., 'images/panorama/*.jpg')

### Results
As demonstrated in the output images:
- Successful feature matching between adjacent images
- Seamless panorama creation
- Green lines showing matched keypoints
- Clean stitching without visible seams

## Parameters and Configuration

### Coin Detection
- Gaussian blur kernel size: (11, 11)
- Canny edge thresholds: (30, 150)
- Minimum contour area: 1000 pixels
- Dilation iterations: 2

### Panorama Stitching
- Lowe's ratio: 0.75
- RANSAC threshold: 4.0
- Feature matching: Brute Force
- Warping method: Perspective

## Observations

### Coin Detection
- Successfully handles non-overlapping coins
- Accurate detection on uniform backgrounds
- Robust to moderate lighting variations
- Clear segmentation boundaries

### Panorama Stitching
- Best results with significant overlap between images
- Handles exposure differences well
- Maintains image quality in the final panorama
- Accurate feature matching as shown by green lines

## Author

AKASH UPADHYAY
Roll No: MT2024013

## License

This project is licensed under the MIT License

