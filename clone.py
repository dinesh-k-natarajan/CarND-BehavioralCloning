## Libraries and Modules
import csv
import cv2
import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

## Helper functions
# Import and augment the images and measurements in the dataset
def createDataset(samples):
    images = []
    measurements = []
    steering_correction = 0.2
    for sample in samples:
        for camera in range(3):
            source_path = sample[camera]
            filename = source_path.split('/')[-1]
            current_path = 'data/IMG/' + filename
            image = cv2.imread(current_path)
            # OpenCV reads image as BGR, converting to YUV (for NVIDIA EEL Model)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            images.append(image)
            # Steering angle correction based on camera position
            if camera == 0:   # center camera
                measurement = float(sample[3])
            elif camera == 1: # left camera
                measurement = float(sample[3]) + steering_correction
            elif camera == 2: # right camera
                measurement = float(sample[3]) - steering_correction
            measurements.append(measurement)
            #  Data Augmentation by flipping the images and steering angles
            images.append(cv2.flip(image,1))
            measurements.append(-1.0*measurement) 
            
    return list(zip(images, measurements)) 
  
                    
# Generator function for the dataset - returns images and measurements based on the paths in samples
def generator(dataset, batch_size=32):
    num_samples = len(dataset)
    while True: # Loop forever so that the generator never terminates
        shuffle(dataset)
        for offset in range(0, num_samples, batch_size):
            batch_samples = dataset[offset:offset+batch_size]
            # Import features(images) and labels(measurements) 
            images = []
            measurements = []
            for image, measurement in batch_samples:
                images.append(image)
                measurements.append(measurement)
                
            # Convert to numpy arrays for Keras compatibility 
            images       = np.array(images)
            measurements = np.array(measurements)
            yield shuffle(images, measurements)

# Neural Network based on NVIDIA End-to-End Learning for Self-Driving Cars
def NVIDIAEEL_Model():
    # Build a Neural Network
    model = Sequential()
    # Preprocessing images - Normalization and Mean Centering
    model.add(Lambda(lambda x: (x/127.5) - 1, input_shape=(160,320,3)))
    # Preprocessing images - Cropping irrelevant features such as hills, hood of the car, etc
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    # Convolutional Layers
    model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(Dropout(0.5))
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50,  activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model

## Body
# Import data from csv file
samples = [] 
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
    
# Create the dataset from the lines of the csv file:
dataset = createDataset(samples[1:])
# Split data into training and validation sets
train_data, validation_data = train_test_split(dataset, test_size=0.2)

# Defining the batch size for training
batch_size = 32
num_epochs = 5

# Using the generator function on the datasets
train_gen      = generator(train_data, batch_size=batch_size) 
validation_gen = generator(validation_data, batch_size=batch_size)

# Neural Network based on NVIDIA End-to-End Learning for Self-Driving Cars
model = NVIDIAEEL_Model()

# Training the model in batches with the help of generator function
model.compile(loss='mse', optimizer = 'adam')
history_object = model.fit_generator(train_gen, \
                                    steps_per_epoch = math.ceil(len(train_data)/batch_size), \
                                    validation_data = validation_gen, \
                                    validation_steps = math.ceil(len(validation_data)/batch_size), \
                                    epochs = num_epochs, verbose = 1)

# Save the model
model.save('model.h5')
print('Model saved!')

# Using the history object to visualize the losses
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation set'], loc='upper right')
plt.savefig('losses.png', bbox_inches='tight')
