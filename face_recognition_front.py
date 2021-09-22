import cv2
import os
import time
from datetime import datetime

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")
recognizer.read('trainer/trainer front.yml')
cascadePath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(1)
duration = 10
start = time.time()
nextTime = start
diffTime = round(nextTime - start, 2)
i = 0

label = input("name: ")
Ids = int(input("Id: "))
vid_name = input("video name: ")
file_name = input("output name: ")

f = open("result/output/" + file_name + ".txt", "w")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result/video/' + vid_name + '.avi', fourcc, 5, (640,480))

print(label, file=f)
print(Ids, file=f)
print("\nstart time : "+str(datetime.now()), file=f)

input("Press Enter to continue...")

while (True and diffTime < duration):
    ret, im =cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3,5)
    
    if i == 0:
        start = time.time()        
        i = i+1
        
    print("start time : "+str(datetime.now()), file=f)
    for(x,y,w,h) in faces:
        #a = Ids
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
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
            #if(Id == 2):
            #   Id = "Raihan {0:.2f}%".format(round(100 - confidence, 2))
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(str_id), (x,y-40), font, 1, (255,255,255), 3)

        
        if Id == Ids:
            print("Result: " + str_id + " - TRUE", file=f)
            print(round(confidence, 2), file=f)
            #print("true :" + str(datetime.now()))
        else :
            print("Result: " + str_id + " - false", file=f)
            print(round(confidence, 2), file=f)
            #print("false :" + str(datetime.now()))            

        out.write(im)
        cv2.imshow('im',im)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
            
    print("end :" + str(datetime.now()), file=f)
    nextTime = time.time()
    diffTime = round(nextTime-start,2)

f.close()
print("\nscript ends")
cam.release()
cv2.destroyAllWindows()
