import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while(1):
    ret, frame = cap.read()
    #print(height)
    #cv2.imshow("Cropped Image", crop_img)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h), (255,0,0), 2)
        #count += 1
        #cv2.imwrite("dataset front/" + name + '.' + ids + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame', gray)
        cv2.waitKey(100)
    #cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
