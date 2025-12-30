📏 Pole Height & Distance Measurement using YOLO and Reference Object
📌 Project Overview

This project detects traffic cones using a YOLO object detection model and uses the known real-world height of the cone (0.7 m) as a reference to estimate:

📐 Pole heights

📏 Distances between selected points

🖱️ Multiple manual measurements per image or video

The system supports images, folders, and videos, and measurements are performed using mouse clicks on the displayed frame.

This project is designed to run locally using Anaconda Prompt.

✨ Key Features

YOLO-based traffic cone detection

Automatic pixel-to-meter scaling using 0.7 m cone height

Manual measurement using mouse clicks

Measure multiple poles in a single image

Measure distance or height

Undo last measurement

Movable floating measurement panel

Works with images and videos

Supports recording output video

Designed for engineering field analysis

🧠 Measurement Logic

YOLO detects traffic cones

Cone pixel height is calculated

Scale is computed:

meters_per_pixel = 0.7 / cone_pixel_height


User clicks:

2 points → one measurement

Each pair is independent

Distance is shown in meters

All measurements are stored and listed

🖱️ Mouse & Keyboard Controls
Mouse
Action	Description
Left Click	Add measurement point
Right Click	Clear all measurements
Keyboard
Key	Action
u	Undo last measurement
c	Clear all measurements
p	Save current annotated image
s	Pause / resume video
q or ESC	Quit
🖥️ Supported Inputs

Single Image (.jpg, .png, .bmp)

Folder of Images

Video (.mp4, .avi, .mkv)

🧰 Environment Setup (Anaconda Prompt)
1️⃣ Create Conda Environment
conda create -n yolo-env1 python=3.9 -y
conda activate yolo-env1

2️⃣ Install Dependencies
pip install ultralytics opencv-python numpy


⚠️ Make sure OpenCV GUI works properly in Anaconda Prompt
(Do not run inside headless terminals)

🚀 Running the Code
Image
python yolo_detect.py --model my_model.pt --source test.jpg

Folder
python yolo_detect.py --model my_model.pt --source ./images/

Video
python yolo_detect.py --model my_model.pt --source video.mp4

Record Output Video
python yolo_detect.py --model my_model.pt --source video.mp4 --record

🏗️ Model Training Pipeline
🔹 Step 1: Labeling (Label Studio)

Install Label Studio:

pip install label-studio


Start Label Studio:

label-studio


Create a project:

Label type: Bounding Boxes

Class name: traffic_cone

Export annotations in YOLO format

🔹 Step 2: Training (Google Colab)

Upload dataset to Google Drive

Open Google Colab

Install YOLO:

!pip install ultralytics


Train:

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640
)


Download best.pt

Use it as --model

📁 Recommended Repository Structure
pole-measurement-yolo/
│
├── yolo_detect.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── images/
│   └── labels/
│
├── model/
│   └── my_model.pt
│
└── samples/
    ├── test.jpg
    └── demo.mp4

📦 requirements.txt
ultralytics
opencv-python
numpy

⚠️ Important Notes

Measurements depend on cone detection accuracy

Camera perspective affects accuracy

Best results when cone and pole are on the same ground plane

Designed for relative measurements, not survey-grade precision

📜 License

This project is for academic and research use.

🙌 Acknowledgements

Ultralytics YOLO

OpenCV

Label Studio

Google Colab
