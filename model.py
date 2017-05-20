import csv
import cv2
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Activation, Dropout, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import History

correction = 0.30

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
