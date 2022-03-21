import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
import cv2
import numpy as np
from PIL import Image
from keras import models
import tensorflow as tf
import pathlib

model = models.load_model('C:\\Users\\Caden\\Desktop\\model.h5')
guesses = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','nothing']

# Used for single image prediction
url = "https://www.signingsavvy.com/images/words/alphabet/2/u1.jpg"
path = tf.keras.utils.get_file('u', url)
# path = pathlib.Path("C:\\Users\\Caden\\Desktop\\asl_test\\Y_test.jpg")

img = tf.keras.utils.load_img(
    path, target_size=(200,200)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
scores = list(zip(guesses, list(predictions[0])))
print(scores)

print("This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(guesses[np.argmax(score)], 100 * np.max(score))
)

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     path2 = "C:\\Users\\Caden\\Desktop\\asl_test\\"
#     _, frame = cap.read()
#     #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame=cv2.flip(frame, 1)
    
#     im = Image.fromarray(frame).resize((200,200))
#     img_array = tf.keras.utils.img_to_array(im)
#     img_array = tf.expand_dims(img_array, 0)

#     predictions = model.predict(img_array)
#     score = tf.nn.softmax(predictions[0])
#     cv2.imshow("Capturing", frame)
#     p = guesses[np.argmax(score)]
#     path2 = cv2.imread(str(path2) + "{}_test.jpg".format(p))
#     cv2.imshow("Prediction", path2)
#     print(p)

#     key=cv2.waitKey(1)
#     if key == ord('q'):
#             break
# cap.release()
# cv2.destroyAllWindows()