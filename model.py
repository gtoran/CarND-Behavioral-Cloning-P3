import csv
import cv2
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Activation, Dropout, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import History

correction = 0.20

# Load Udacity's driving log data
lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

# Build images & measurement array
for line in lines:
    # Let's consider all camera angles
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        source_path = 'data/IMG/'+filename
        image = cv2.imread(source_path)
        images.append(image)

    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement + correction)
    measurements.append(measurement - correction)

# Data augmentation - let's "drive" in the opposite direction by flipping all images and measurements
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)
 
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# LeNet
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
exit()
