# **Webcam Vehicle Detection on Raspberry Pi**

Description of Repository
=========================
This repository contains code and setup instructions for running vehicle-only object detection from a webcam feed on Raspberry Pi.

Details of Software and Neural Network Model for Object Detection:
* Language: Python
* Framework: TensorFlow Lite
* Network: SSD MobileNet-V2
* Training Dataset:Berkely Deep Drive (BBD100K)


The motivation for the Project
========================
The goal is to run a lightweight TensorFlow Lite detector on Raspberry Pi and only render vehicle detections (car, bus, truck, motor/bike, train) from a live webcam stream.


Additional Resources
===================
* **YouTube Turorial For This Repository**: https://youtu.be/Zfmo3bMycUg
* **Post Describing the Training Procedure**: https://ecd1012.medium.com/autonomous-driving-object-detection-on-the-raspberry-pi-4-175bba51d5b4
* **Explanation of Machine Learning/Deep Learning/Object Detection**: https://www.youtube.com/watch?v=pIciURImE04&t=138s&ab_channel=bitsNblobsElectronics

Source
=======
**Reference for Source Code for the Project**: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md

Special thanks to Evan from EdjeElectronics for the instructions and the majority of the code used in this project! :)

Results
=======
<img src="images/result.gif" width="800" height="600">

Vehicle Testing Configuration
=============================
<img src="images/rpi_setup.jpg" width="400" height="300">

Core
* Raspberry Pi 4 GB
* USB webcam or Pi camera exposed as `/dev/video*`
* 3.5 Amp USB-C Power Supply

This tissue box setup isn't the greatest, but it's what I used to mount the PI on the dashboard of my car. I then used the USB-C cable plugged into the AC outlet of my car while I drove around to record and process footage.

Issues
======
1.) If you get an error when trying to run the program showing the following: 
```
ImportError: No module named cv2
```
Try using this tutorial to install and build opencv: https://pimylifeup.com/raspberry-pi-opencv/
The software setup steps should install OpenCV, but sometimes installing it on the Raspberry Pi can be finicky.


Setting Up Software
====================
1.) Clone Repository:
```
git clone https://github.com/ecd1012/rpi_road_object_detection.git
````
2.) Change directory to source code:
```
cd rpi_road_object_detection
```
3.) Open command prompt and make sure Pi is up to date:
```
sudo apt-get update && sudo apt-get upgrade
```
4.) Make virtual environment:
```
python3 -m venv .venv
```
5.) Activate environment:
```
source .venv/bin/activate
```
6.) Install dependencies:
```
bash get_py_requirements.sh
```
7.) Plug in your webcam (or enable Pi camera if using a V4L2 bridge).



Running Detection
=================
8.) Run detection:
```
python TFLite_detection_webcam_loop.py --modeldir=TFLite_model_bbd
```
The script opens the webcam, runs TensorFlow Lite inference, and draws only vehicle detections. Press `q` to quit.

The default detection zone is a fixed green box in the lower-middle of the frame. When a
vehicle detection's center point enters that zone, the `Vehicle detected` sign stays on for
5 seconds. Each new hit inside the zone resets the timer back to 5 seconds.

Zone timing flags:
```
# Keep the default 5-second hold and zone position
python TFLite_detection_webcam_loop.py --modeldir=TFLite_model_bbd

# Custom hold time in seconds
python TFLite_detection_webcam_loop.py --modeldir=TFLite_model_bbd --hold-seconds=5

# Custom zone as normalized x1,y1,x2,y2
python TFLite_detection_webcam_loop.py --modeldir=TFLite_model_bbd --roi=0.35,0.55,0.65,0.95
```

Optional camera flags:
```
# USB webcam (default)
python TFLite_detection_webcam_loop.py --modeldir=TFLite_model_bbd --camera-source=usb --camera-id=0

# CSI camera (Picamera2/libcamera)
python TFLite_detection_webcam_loop.py --modeldir=TFLite_model_bbd --camera-source=csi

# Quick CSI test (no detection), uses rpicam-hello
python TFLite_detection_webcam_loop.py --modeldir=TFLite_model_bbd --test-csi --test-csi-seconds=8
```

Running on Boot
===============
9.) Use `systemd` so detection starts automatically when the Pi boots:
```
chmod +x start_on_boot.sh
sudo cp rpi-road-object-detection.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rpi-road-object-detection.service
sudo systemctl start rpi-road-object-detection.service
```

By default the boot launcher uses USB camera. To switch the boot service to CSI:
```
sudo systemctl edit rpi-road-object-detection.service
```
Add:
```
[Service]
Environment=CAMERA_SOURCE=csi
```
Then reload and restart:
```
sudo systemctl daemon-reload
sudo systemctl restart rpi-road-object-detection.service
```

10.) Check service status/logs:
```
sudo systemctl status rpi-road-object-detection.service
journalctl -u rpi-road-object-detection.service -f
```

11.) Optional service controls:
```
sudo systemctl restart rpi-road-object-detection.service
sudo systemctl stop rpi-road-object-detection.service
sudo systemctl disable rpi-road-object-detection.service
```

If no window appears after boot:
```
sudo systemctl status rpi-road-object-detection.service
journalctl -u rpi-road-object-detection.service -n 100 --no-pager
```
Also make sure you are booting into Raspberry Pi Desktop (not headless console-only mode).









