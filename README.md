
# Pole Height Measurement Using Traffic Cone Reference (YOLO + OpenCV)

## 📌 Project Overview

This project measures **telecom pole heights in meters** using **images or videos** by referencing a **traffic cone of known height (0.7 m)** detected via a YOLO model.

Instead of detecting poles using AI, the system allows the **user to manually select pole bottom and top points using the mouse**, which improves flexibility and accuracy in real-world scenarios.

The workflow combines:
- YOLO-based **traffic cone detection**
- Pixel-to-meter scale calculation
- Mouse-based pole height measurement
- Support for **multiple poles per image**
- Undo, save, and GUI display features





## 🎯 Features

- Detect **traffic cones** using YOLO
- Use cone height as **0.7 m reference**
- Calculate **pixel-to-meter ratio**
- Measure **multiple poles** in a single image
- Works with **images and videos**
- Interactive **mouse-based point selection**
- Display height with **arrows + meter values**
- Movable **GUI data panel**
- Undo last measurement
- Save all pole heights
- Runs in **Anaconda Prompt**

## 🧠 Measurement Logic

1. YOLO detects a traffic cone
2. Pixel height of cone is measured
3. Scale is computed:
meters_per_pixel = 0.7 / cone_pixel_height

4. User clicks:
- Bottom of pole
- Top of pole
5. Pole height is calculated as:
pole_height_m = pixel_distance × meters_per_pixel

Each pole is treated **independently**.


## 🛠️ Technologies Used

- Python 3.9+
- OpenCV
- Ultralytics YOLO
- NumPy
- Label Studio
- Google Colab
- Anaconda


## 🧪 Dataset Preparation (Label Studio)

### Step 1: Install Label Studio
```bash
pip install label-studio
label-studio start
```

### Step 2: Label Traffic Cones

- Create Object Detection project
- Label class:
traffic_cone
- Export annotations in YOLO format

### 🚀 Model Training (Google Colab)

1. Upload dataset to Google Drive

2. Use Ultralytics YOLO

3. Train model

4. Download best.pt

5. Rename to my_model.pt

## ▶️ How to Run (Anaconda Prompt)
### Step 1: Activate Environment
```bash
conda activate yolo-env1
```
### Step 2: Run on Image
```bash 
python yolo_detect.py --model "path/to/my_model.pt" --source "path/to/image.jpg"
```
### Step 3: Run on Video
```bash
python yolo_detect.py --model "path/to/my_model.pt" --source "path/to/video.mp4"
```

## 🖱️ Mouse Controls

| Action         | Function           |
| -------------- | ------------------ |
| Left Click (1) | Select pole bottom |
| Left Click (2) | Select pole top    |
| `u`            | Undo last pole     |
| `h`            | Toggle height info |
| `d`            | Toggle detection   |
| `q`            | Quit               |

## 🧾 GUI Panel

- Displays:

    - Pole number

    - Height in meters

- Panel is draggable

- Updates live as poles are added

- All measurements remain visible   

## 🧠 Design Decisions

- Manual pole selection chosen over AI detection for:

    - Better accuracy

    - Different pole types

    - Occlusions

- Traffic cone chosen as reference due to:

    - Standardized height (0.7 m)

    - Common roadside availability

## 📌 Limitations

- Requires visible traffic cone

- Accuracy depends on perspective

- Camera must remain static per measurement

## 📈 Future Improvements

- Automatic pole detection

- Perspective correction

- Depth estimation

- Mobile app version

- Export measurements to CSV

## 👤 Author

**Iwanthi Harshani** 
Final Year / Research-Oriented Computer Vision Project
Sri Lanka
