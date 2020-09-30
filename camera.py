import cv2
import numpy as np
import tensorflow as tf
import keras
from keras.applications import imagenet_utils

labels = ["Roadworks" , "Giveway" , "Speed-50" , "Speed-60"]
model = tf.keras.models.load_model("traffic_Sign.model")
cap=cv2.VideoCapture(0)
if(cap.isOpened()):
    print("Camera OK")
else:
    cap.open()
while True:
    ret, original = cap.read()
    frame = cv2.resize(original,(64,64))
    image= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = image.reshape((1,)+ image.shape)
    image = imagenet_utils.preprocess_input(image)
    preds= model.predict(image,batch_size=1)
    pred= model.predict_classes(image,  batch_size=1)
    if(max(preds[0])*100 > 50):
        print(max(preds[0])*100,'****',labels[int(pred)])
        cv2.putText(original, "Label: {}".format(labels[int(pred)]), (10,30),cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 255, 0))
        cv2.imshow("Classificaiton", original)
        if(cv2.waitKey(1) & 0xFF==ord('q')):
            break
    else:
        print(max(preds[0])*100,'****',"Recognition Failure...")
        cv2.putText(original, "Label: {}".format('Recongnition Failed'), (10,30),cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 255, 0))
        cv2.imshow("Classificaiton", original)
        if(cv2.waitKey(1) & 0xFF==ord('q')):
            break
cap.realse()
cv2.destroyAllWindows()