import numpy as np # Math
import pandas as pd # Read and update data
import csv # Read csv files
import cv2 # OpenCV2 for image processing
# Splits data into training and test sets
import sklearn
from sklearn.model_selection import train_test_split
# Shuffle order of training set each epoch
from sklearn.utils import shuffle
# Keras is a high-level wrapper (library) around TensorFlow
# Sequential is linear stack of layers
from keras.models import Sequential
# Model layers
from keras.layers import Cropping2D, Dense, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
import argparse # Reading command line arguments
import os # Reading files

# Create list of lists of data and corresponding images
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split to training/validation sets 80%/20%; Leave out samples[0], which are column labels.
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)


def generator(samples, batch_size=32):
    """
    Generator allows for large amounts of data to be created and iterated over
    in small batches, as opposed for waiting for it to go through all data at once.
    Args:
        samples: data to process (list)
        batch_size: samples processed per generation
    Yields (returns):
        X_train: shuffled numpy array of training images with added flipped images
        y_train: shuffled numpy array of steering angles with added flipped steering anglesß
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        # Iterate through range counting by batch_size
        for offset in range(0, num_samples, batch_size):
            # Equals batch_size-length array of samples
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                # Takes center image label substring we initially took from csv
                # Attach path to name to call corresponding image file
                name = './data/IMG/' + batch_sample[0].split('/')[-1]

                # Create OpenCV image file from full path in name
                center_image = cv2.imread(name)

                # Convert BGR to RGB color for Keras processing
                center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

                # Corresponding steering angle at index 3
                center_angle = float(batch_sample[3]) # Convert to float

                # Add image and angle to same index in seperate arrays
                images.append(center_image_rgb)
                angles.append(center_angle)

                # Add flipped version of image and angle to augment data set
                # Evens left and right turns, reducing overfitting
                images.append(cv2.flip(center_image_rgb, 1))
                angles.append(-float(line[3]))

            # Convert images and angles to numpy arrays for Keras
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train) # Yield/return shuffled batch

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 160, 320, 3  # Trimmed image format

model = Sequential() # Linear stack of layers
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x / 127.5 - 1.,input_shape=(ch, row, col),output_shape=(ch, row, col)))
# Crops image ((top_crop, bottom_crop), (left_crop, right_crop))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# 5x5 convolution with 24 filters, 2x2 stride (subsample), relu activation
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
# 5x5 convolution with 36 filters, 2x2 stride (subsample), relu activation
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
# 5x5 convolution with 48 filters, 2x2 stride (subsample), relu activation
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
# 3x3 convolution with 64 filters, relu activation
model.add(Convolution2D(64, 3, 3, activation='relu'))
# 3x3 convolution with 64 filters, relu activation
model.add(Convolution2D(64, 3, 3, activation='relu'))
# Compress tensor to single vector
model.add(Flatten())
# Fully connected layer with output of 100
model.add(Dense(100))
# Fully connected, output 50
model.add(Dense(50))
# Fully connected, output 10
model.add(Dense(10))
# Fully connected, output 1
model.add(Dense(1))

# Print summary of layers
model.summary()
# Use mean squared error and Adam optimizer
model.compile(loss='mse', optimizer='adam')
# Fit model using generator for training and validation, 10 epochs
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

# Save model to local folder
model.save('model.h5')

'''

angle_adjustment = 0.1

def img_process(X, y, line):
    """
    Args: X (list of list of images), y (list of steering angles), line (list/csv line), cam_view_index (int).
    Returns: images and angles with original and flipped images, converted to RGB.
    """
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename

        image = cv2.imread(current_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        X.append(image_rgb)
        y.append(float(line[3]))

        # Flips image
        X.append(cv2.flip(image_rgb, 1))
        angle_adjustment = 0
        if i > 0:
            angle_adjustment = (-1) ** (i - 1)
        # Flips angle relative to flipped image
        y.append(-float(line[3]) + angle_adjustment)

    return X, y

lines = []
images = []
angles = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

    for line in lines[1:]:
        #print(line)
        center_image = cv2.imread('./data/IMG/' + line[0].split('/')[-1])
        center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

        images.append(center_image_rgb)
        angles.append(float(line[3]))
        #flipped
        images.append(cv2.flip(center_image_rgb, 1))
        angles.append(-float(line[3]))

        left_image = cv2.imread('./data/IMG/' + line[1].split('/')[-1])
        left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

        images.append(left_image_rgb)
        angles.append(float(line[3])+angle_adjustment)
        #flipped
        images.append(cv2.flip(left_image_rgb, 1))
        angles.append(-(float(line[3])+angle_adjustment))

        right_image = cv2.imread('./data/IMG/' + line[2].split('/')[-1])
        right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

        images.append(right_image_rgb)
        angles.append(float(line[3])-angle_adjustment)
        #flipped
        images.append(cv2.flip(right_image_rgb, 1))
        angles.append(-(float(line[3])-angle_adjustment))

'''
