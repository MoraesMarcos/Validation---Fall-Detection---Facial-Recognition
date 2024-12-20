import cv2
import os

cam = cv2.VideoCapture(1)
cam.set(3, 1920) 
cam.set(4, 1080)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\nDigite o ID do usuário e pressione <enter>: ')
print("\n [INFO] Inicializando captura de rosto. Olhe para a câmera e aguarde ...")
count = 0

while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    elif count >= 1000: 
         break

print("\n [INFO] Programa encerrado. Limpando recursos...")
cam.release()
cv2.destroyAllWindows()