import os
import numpy as np
import cv2
import keras
import pandas as pd
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential

data_dir = os.path.abspath('C:\\')
list_images = []
output = []
for dir in os.listdir(data_dir):
    inner_dir = os.path.join(data_dir, dir)
    csv_file = pd.read_csv(os.path.join(inner_dir,"GT-" + dir + '.csv'), sep=';')
    for row in csv_file.iterrows() :
        img_path = os.path.join(inner_dir, row[1].Filename)
        img = cv2.imread(img_path)
        img = img[row[1]['Roi.X1']:row[1]['Roi.X2'],row[1]['Roi.Y1']:row[1]['Roi.Y2'],:]
        img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_LINEAR)
        list_images.append(img)
        output.append(row[1].ClassId)
input_array = np.stack(list_images)
train_y = keras.utils.np_utils.to_categorical(output)
randomize = np.arange(len(input_array))
np.random.shuffle(randomize)
x = input_array[randomize]
y = train_y[randomize]
split_size = int(x.shape[0]*0.6)
train_x, val_x = x[:split_size], x[split_size:]
train1_y, val_y = y[:split_size], y[split_size:]
split_size = int(val_x.shape[0]*0.5)
val_x, test_x = val_x[:split_size], val_x[split_size:]
val_y, test_y = val_y[:split_size], val_y[split_size:]
model = Sequential([
 Conv2D(32, (3, 3), activation='relu', input_shape=(64,64,3), padding='same'),
 BatchNormalization(),
 Conv2D(64, (3, 3), activation='relu', padding='same'),
 MaxPooling2D(pool_size=(2, 2), strides=1),
 Dropout(0.2),
 Conv2D(64, (3, 3), activation='relu', padding='same'),
 MaxPooling2D(pool_size=(2, 2), strides=1),
 Flatten(),
 Dense(units=100, activation='relu'),
 Dropout(0.5),
 Dense(units=4, activation='softmax')])
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
trained_model_conv = model.fit(train_x.reshape(-1,64,64,3), train1_y, epochs=24, batch_size=16, validation_data=(val_x, val_y))
model.save('traffic_Sign.model')
pred = model.predict_classes(test_x)
loss , accuracy = model.evaluate(test_x, test_y)
print( "Loss of {}".format( loss ) , "Accuracy of {} %".format( accuracy * 100 ) )
