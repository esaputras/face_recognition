import cv2
import os

vid_cam = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

while(True):
    _, image_frame = vid_cam.read()
    gray = cv2.cvtColor (image_frame, cv2.COLOR_BGR2GRAY)

    right = cv2.flip(gray,1)
    faces = faceCascade.detectMultiScale(right, 1.3, 5)

    for (x,y,w,h) in faces :
        cv2.rectangle(image_frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        cv2.imshow('im', 
