import numpy as np # Math
import pandas as pd # Read and update data
import csv # Read csv files
import cv2 # OpenCV2 for image processing
# Splits data into training and test sets
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

import os
import csv

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=3)

path = '/Users/tomdunlap/projects/udacity/sd_car/CarND-Behavioral-Cloning-P3/data/'
angle_adjustment = 0.1

'''
#for debugging, allows for reproducible/deterministic results
np.random.seed(0)

def load_data(args):
    """
    Load training data and split into training and testing sets
    """
    # Reads CSV file into a single dataframe
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))
    # Images are inputs
    X = data_df[['center', 'left', 'right']].values
    # Steering angle is output data
    y = data_df['steering'].values

    # 80/20 split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y test_size=args.test_size, random_state=0)

    return X_train, X_test, y_train, y_test
'''

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
        # Flips angle relative to flipped image
        y.append(-float(line[3]))

    return X, y

'''
for line in reader:
    for i in range(3):
        images, angles = img_process(images, angles, line, i)
'''

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
'''
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


'''









print(lines[2])
for line in lines[1:]:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename

        image = cv2.imread(current_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images.append(image_rgb)
        angles.append(float(line[3]))

        # Flips image
        images.append(cv2.flip(image_rgb, 1))
        # Flips angle relative to flipped image
        angles.append(-float(line[3]))


'''


'''
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    angle = float(line[3])
    angles.append(angle)
'''
'''
        print

'''
images = np.array(images)
angles = np.array(angles)

# Split data to training/test : 80%/20%
X_train, X_test, y_train, y_test = train_test_split(images, angles, test_size=0.2)

'''
# Convert to numpy for Keras
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
'''

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, shuffle=True)

model.save('model_mine.h5')
