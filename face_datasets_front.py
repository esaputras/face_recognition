import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
 
vid_cam = cv2.VideoCapture(1)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
assure_path_exists("dataset/front dataset/")

while(True):
    name = input("Siapa? ")
    count = 0
    if (name == "q"):
        break
    ids = input("id? ")
    while(True):        
        _, image_frame = vid_cam.read()
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(image_frame,(x,y),(x+w,y+h), (255,0,0), 2)
            count += 1
            cv2.imwrite("dataset/front dataset/" + name + '.' + ids + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('frame', image_frame)
            cv2.waitKey(100)
            print (count)
        if(count > 39):
            break
        
        
vid_cam.release()
cv2.destroyAllWindows()
