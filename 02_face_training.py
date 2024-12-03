import cv2
import numpy as np
import os 
from PIL import Image

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # Abrir imagem e converter para escala de cinza
        img_numpy = np.array(PIL_img, 'uint8')  # Converter imagem para matriz numpy
        id = int(os.path.split(imagePath)[-1].split(".")[1])  # Extrair ID do arquivo
        faces = faceCascade.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

# Configurações iniciais
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Obter faces e IDs para treinamento
faces, ids = getImagesAndLabels(path)

# Treinar o reconhecedor
print("\n [INFO] Treinando o reconhecedor de faces. Isso pode levar alguns segundos. Aguarde ...")
recognizer.train(faces, np.array(ids))

# Salvar o modelo treinado
recognizer.write('trainer/trainer.yml')

print("\n [INFO] Treinamento concluído. {0} rostos treinados.".format(len(np.unique(ids))))