#IDs for landmarks (documentation):
#https://google.github.io/mediapipe/solutions/hands.html

import cv2
import mediapipe as mp
import time
import math

#images
import cv2
import urllib.request
import numpy as np
import ssl

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#distance formula:
def distance(x1, x2, y1, y2):
    formula=math.sqrt( ((x2-x1)**2) + ((y2-y1)**2) )
    #we can include a summation formula to add up the distances FOR EACH LANDMARK
    return formula


url='https://alphabetizer.flap.tv/lists/images/American_Sign_Language_Chart.jpg'
req = urllib.request.urlopen(url, context=ssl._create_unverified_context())
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1) # 'Load it as it is'
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#img=img.resize((300,300))

# For webcam input:
cap = cv2.VideoCapture(0)
hands=mp_hands.Hands()
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    #****Line below places text over image, can be duplicated
    #image = cv2.putText(image, 'o', #This is the string displayed
     #                   org=(200,200), #coordinates (cx,cy)
      #                  fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,color=(0,0,255), thickness=2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h,w,c = image.shape
                #dimensions:
                #h=480   w=640     c=3
                #middle: (240,320)

                cx, cy = int(lm.x*w), int(lm.y * h)
                ids = [4]
                #print("h: ",h, "w: ",w, "c: ",c, "cx :", cx, "cy: ",cy)
                if id in ids:
                    # Roughly the bottom left corner
                    #if((400 < cx) and (400 < cy)):
                        #print(cx, cy, lm.x, lm.y)
                        print(1/distance(lm.x,.75, lm.y, .75))
                        #score=distance(cx,499,cy,499)
                        #print("spot! score:")
                        #print(score)
                        
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # Flip the image horizontally for a selfie-view display.
    image=cv2.flip(image, 1)
    output = cv2.addWeighted( image[90:390,170:470,:], 0.7, img[150:450,100:400,:], 0.3, 0)
    # gamma is the scalar added to each sum
    image[90:390,170:470,:] = output
    cv2.imshow('MediaPipe Hands',image) # cv2.flip(image, 1)
    if (cv2.waitKey(1) & 0xFF == ord('q')) or(cv2.waitKey(1)%256 == 27): #HOLD ESC
        break
print('Game Over!')
cap.release()
cv2.destroyAllWindows()