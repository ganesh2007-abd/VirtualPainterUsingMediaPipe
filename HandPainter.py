import cv2 as cv
import mediapipe as mp
import time
import HandDetectorModule as hdm
import numpy as np
import os


folderpath = "Headers"
mylist = os.listdir(folderpath)
overlaylist = []
for impath in mylist:
    image = cv.imread(f'{folderpath}/{impath}')
    image = cv.resize(image,(1280,125),interpolation=cv.INTER_AREA)
    overlaylist.append(image)

cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = hdm.handDetector(trackcon=0.75)
header = overlaylist[0]

drawcolor = (0,0,255)

while True:
    success,img = cap.read()
    if not success:
        break
    img = cv.flip(img,1)
    
    img = detector.findHands(img)
    lmlist = detector.findposition(img,False)

    if len(lmlist) != 0:
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

        fingers = detector.fingerUp()
        # print(fingers)

        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            cv.rectangle(img,(x1,y1-25),(x2,y2+25),drawcolor,cv.FILLED)
            if y1<140:
                if 0 < x1 < 250:
                    header = overlaylist[1]
                    drawcolor = (0,0,255)
                elif 300<x1<550:
                    header = overlaylist[2]
                    drawcolor = (0,255,0)
                elif 600<x1<850:
                    header = overlaylist[3]
                    drawcolor = (255,0,0)
                elif 900 < x1 < 1150:
                    header = overlaylist[4]
                    drawcolor = (0,0,0)
                
    img[0:125,0:1280] = header

    cv.imshow("Image",img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()