
import numpy as np                                                                      #For working with matrices
import cv2
import pickle
face_cascade = cv2.CascadeClassifier('Cascades\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {"person_name":1}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}#Creates an object for face detection using the trained xml file
#eye_cascade = cv2.CascadeClassifier('Cascades\haarcascade_eye.xml')                     #Creates an object for eye detection using the trained xml file
cap = cv2.VideoCapture(0)                                                               #Creates an object to obtain images from laptop webcam
while 1:                                                                                #Infinite loop until broken by keyboard interrupt
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                                          #Obtain image from camera
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=3)                                         #Obtain a list of coordinates of rectangles covering the faces
    #print(faces)
    for (x,y,w,h) in faces:                                                             #For each rectangle covering the faces in the image
                                                                                        #Draw a green rectanlge using the coordinates
        roi_gray = gray[y:y+h, x:x+w]                                                   #Cut the region of interest from the whole image
        roi_color = gray[y:y+h, x:x+w]
        id_,conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=200:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0,0,255)
            stroke = 2
            cv2.putText(img,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0,0,255)
            stroke = 2
            cv2.putText(img,"Unknown",(x,y),font,1,color,stroke,cv2.LINE_AA)
        img_item = "boom.png"
        cv2.imwrite(img_item,roi_color)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)                                #Obtain the colored region of interest
        #eyes = eye_cascade.detectMultiScale(roi_gray)                                   #Obtain a list of coordinates of rectangles covering the eyes
        #for (ex,ey,ew,eh) in eyes:                                                      #For each rectangle covering the eyes in the image
         #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)                  #Draw the rectanlge
                                                                                        #
    cv2.imshow('img',img)                                                               #Display the final image
    k = cv2.waitKey(30)                                                                 #Keyboard interrupt
    if k == 97:                                                                         #If escape key is pressed all windows gets closed
        break                                                                           #
                                                                                        #
cap.release()                                                                           #
cv2.destroyAllWindows()                                                                 #
#########################################################################################

