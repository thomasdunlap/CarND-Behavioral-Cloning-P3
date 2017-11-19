import numpy as np # read and update dataframes
import pandas as pd # math
# splits data into training and test sets
from sklearn.model_selection import train_test_split
# keras is a high-level wrapper (library) around TensorFlow
# Sequential is linear stack of layers
from keras.models import Sequential
# optimizer using gradient descent
from keras.optimizers import Adam
# save model checkpoints to load later
from keras.callbacks import ModelCheckpoint
# model layers
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
# helper class to define input shape and generate training images given
# image paths and steering angles
from utils import INPUT_SHAPE, batch_generator
# for command line arguments
import argparse
# for reading files
import os

#for debugging, allows for reproducible/deterministic results
np.random.seed(0)

def load_data(args):
    """
    Load training data and split into training and testing sets
    """
    #reads CSV file into a single dataframe
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))
    #images are inputs
    X = data_df[['center', 'left', 'right']].values
    # steering angle is output data
    y = data_df['steering'].values

    # 80/20 split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y test_size=args.test_size, random_state=0)

    return X_train, X_test, y_train, y_test

def build_model(args):
    pass


def main():
    """
    Load and train model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')
