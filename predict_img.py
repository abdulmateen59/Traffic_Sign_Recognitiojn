import cv2
import numpy as np
import tensorflow as tf
import keras

model = tf.keras.models.load_model("traffic_Sign.model")
labels = ["Roadworks" , "Giveway" , "Speed-50" , "Speed-60"]

print("**************************************")

image = cv2.imread("00004_00027.ppm") #Giveway
image = cv2.resize(image,(64,64))
image= image/255
img=image.reshape(-1,64,64,3)
img = np.array(img).astype(np.float32)
pred= model.predict(img)
x=list(np.int_(pred[0]*100))
print(x)




import random
import numpy as np

WINDOW_SIZES = [i for i in range(20, 160, 20)]


def get_best_bounding_box(img, predict_fn, step=10, window_sizes=WINDOW_SIZES):
    best_box = None
    best_box_prob = -np.inf

    # loop window sizes: 20x20, 30x30, 40x40...160x160
    for win_size in window_sizes:
        for top in range(0, img.shape[0] - win_size + 1, step):
            for left in range(0, img.shape[1] - win_size + 1, step):
                # compute the (top, left, bottom, right) of the bounding box
                box = (top, left, top + win_size, left + win_size)

                # crop the original image
                cropped_img = img[box[0]:box[2], box[1]:box[3]]

                # predict how likely this cropped image is dog and if higher
                # than best save it
                print('predicting for box %r' % (box, ))
                box_prob = predict_fn(cropped_img)
                if box_prob > best_box_prob:
                    best_box = box
                    best_box_prob = box_prob

    return best_box

def predict_function(x):
    # example of prediction function for simplicity, you
    # should probably use `return model.predict(x)`
    random.seed(x[0][0])
    return random.random()


# dummy array of 256X256
img = np.arange(256 * 256).reshape((256, 256))

best_box = get_best_bounding_box(img, predict_function)
print('best bounding box %r' % (best_box, ))