import cv2
import numpy as np
import tensorflow as tf
import keras

labels = ["Speed-50","Speed-60","Road-Narrows","Construction-Site"]
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
    preds= model.predict(image)
    pred= model.predict_classes(image)
    if(max(preds[0])*100 > 70):
        print(max(preds[0])*100,'****',labels[int(pred)])
        cv2.putText(original, "Label: {}".format(labels[int(pred)]), (10,30),cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 255, 0))
        cv2.imshow("Classificaiton", original)
        if(cv2.waitKey(1) & 0xFF==ord('q')):
            break
    else:
        print("Recognition Failure...")
        cv2.putText(original, "Label: {}".format('Recongnition Failed'), (10,30),cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 255, 0))
        cv2.imshow("Classificaiton", original)
        if(cv2.waitKey(1) & 0xFF==ord('q')):
            break
cap.realse()
cv2.destroyAllWindows()