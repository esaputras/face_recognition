import cv2
import time
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer2 = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trained left profile.yml')
recognizer2.read('trainer/trained right profile.yml')

faceCascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

vid_cam = cv2.VideoCapture(0)
duration = 10
start = time.time()
nextTime = start
diffTime = round(nextTime - start, 2)
i = 0

label = input("name: ")
position = input("position: ")
Ids = int(input("Id: "))
vid_name = input("video name: ")
file_name = input("output name: ")

f = open("result/output/" + file_name + ".txt", "w")
print(label, file=f)
print(position, file=f)
print(Ids, file=f)
print("start time : "+str(datetime.now()), file=f)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result/video/' + vid_name + '.avi', fourcc, 5, (640,480))

input("Press Enter to continue...")

while (True and diffTime < duration):
    _, im = vid_cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    if (position == 'kiri'):
        faces = faceCascade.detectMultiScale(gray, 1.3,5)
    else:
        right_gray = cv2.flip(gray, 1)
        faces = faceCascade.detectMultiScale(right_gray, 1.3, 5)

    if i == 0:
        start = time.time()        
        i = i+1
        
    print("start time : "+str(datetime.now()), file=f)
    for(x,y,w,h) in faces:
        right_im = cv2.flip(im, 1)
        if(position == 'kiri'):
            cv2.rectangle(im,(x-20,y-20), (x+w+20,y+h+20), (0,255,0), 2)
            Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        else:
            cv2.rectangle(right_im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 2)
            Id, confidence = recognizer2.predict(right_gray[y:y + h, x:x + w])

        if(Id==1 and confidence < 110):
            str_id = "Bunga"
        elif(Id==2 and confidence < 110):
            str_id="Dhifa"
        elif(Id==3 and confidence < 110):
            str_id="Endang"
        elif(Id==4 and confidence < 110):
            str_id="Hasanah"
        elif(Id==5 and confidence < 110):
            str_id="Bulan"
        elif(Id==6 and confidence < 110):
            str_id="Esa"
        elif(Id==7 and confidence < 110):
            str_id="Satria"
        elif(Id==8 and confidence < 110):
            str_id="Ashilla"
        elif(Id==9 and confidence < 110):
            str_id="Wildan"
        elif(Id==10 and confidence < 110):
            str_id="Raihan"
        else:
            str_id = "unknown"            
        
        if(position == "kiri"):
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, str(str_id), (x,y-40), font, 1, (255,255,255), 3)
            out.write(im)
            cv2.imshow('im',im)
        else:
            cv2.rectangle(right_im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(right_im, str(str_id), (x,y-40), font, 1, (255,255,255), 3)
            out.write(right_im)
            cv2.imshow('im',right_im)
        
        if Id == Ids:
            print("Result: " + str_id + " - TRUE", file=f)
            print(round(confidence, 2), file=f)   
        else:
            print("Result: " + str_id + " - FALSE", file=f)
            print(round(confidence, 2), file=f)
     
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
            
    print("end :" + str(datetime.now()), file=f)
    nextTime = time.time()
    diffTime = round(nextTime-start,2)

f.close()
print("\nscript ends")
vid_cam.release()
cv2.destroyAllWindows()
