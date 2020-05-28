# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## **Behavioral Cloning of Lateral Control using NVIDIA's End-to-End Learning Model**

#### In this project, a simplified version of NVIDIA's [End-to-End Learning Model for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) is used to automate lateral control of a vehicle. Training Data Collection and Testing of the Trained Model is done via [Udacity and Unity's Self-Driving Car Simulator.](https://github.com/udacity/self-driving-car-sim)
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Using the simulator to collect data of good driving behavior
* Building a convolution neural network in Keras that predicts steering angles from images
* Training and validating the model with a training and validation set
* Testing that the model successfully drives around track one without leaving the road
* Summarizing the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---  

### Important Contents of the repository

* `data/`     - Directory containing the recorded driving behavior using the simulator in training model
* `clone.py`  - Script for implementation and training of the model using Keras
* `model.h5`  - Trained model
* `drive.py`  - Script for driving the car autonomously in the simulator using the trained model
* `video.mp4` - Video recording of the car driving autonomously using the trained model 

### Instructions for Use

1. [Udacity's Simulator](https://github.com/udacity/self-driving-car-sim) can be cloned from GitHub
2. The implemented model in clone.py is trained and saved as `model.h5` using:
    ```sh
    python clone.py 
    ```
3. The trained model `model.h5` is loaded into the simulator environment using `drive.py`:
    ```sh
    python drive.py model.h5 
    ```
4. Additionally, screenshots of the car in the simulator's autonomous mode can be recorded using:
    ```sh
    python drive.py model.h5 video
    ```
5. Finally, a video can be created using the recorded screenshots using:
    ```sh
    python video.py video
    ```
---

## 1. Dataset
### 1.1 About the dataset

The dataset contains recordings of driving behavior in Track 1 of Udacity's Simulator Environment. 

The dataset contains the following features of the driving behavior at a given time instant:

* Left Camera Image
* Center Camera Image
* Right Camera Image
* Steering Angle
* Throttle
* Braking
* Speed

Since, the main focus is on lateral control, only the three types of images and steering angle measurements are used by the model. The `driving_log.csv` contains the dataset (with paths to the images) and the images are found in `IMG/` directory.  

### 1.2 

