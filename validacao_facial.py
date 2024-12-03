import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def reconhecer_e_coletar_dados():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    names = ['Desconhecido', 'Marcos' , 'Mariana', 'z', 'W']  # IDs reais e nomes associados

    ids_reais = []  # IDs reais informados pelo usuário
    ids_previstos = []  # IDs previstos pelo modelo

    cam = cv2.VideoCapture(1)
    cam.set(3, 1920)  # Reduzindo a resolução para desempenho
    cam.set(4, 1080)

    print("[INFO] Use as teclas (0-9) para informar o ID real ou 'ESC' para sair.")
    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERRO] Não foi possível acessar a câmera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            id_previsto, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Determinar se o rosto é reconhecido ou desconhecido
            if confidence < 35:
                name = names[id_previsto]
            else:
                id_previsto = 0  # ID para desconhecido
                name = "Desconhecido"

            # Desenhar a caixa ao redor do rosto e exibir o nome
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Reconhecimento', img)

        # Obter tecla pressionada
        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # Pressionar ESC para sair
            print("[INFO] Encerrando o programa.")
            break
        elif 48 <= key <= 57:  # Teclas 0-9 para IDs reais
            id_real = key - 48  # Converte para inteiro (0 para "Desconhecido", 1-9 para IDs conhecidos)
            ids_reais.append(id_real)
            ids_previstos.append(id_previsto)
            print(f"[INFO] ID real informado: {id_real}, ID previsto: {id_previsto}")

        if len(ids_reais) >= 100:  # Limitar o número de amostras para simplificação
            print("[INFO] Número máximo de amostras atingido. Encerrando.")
            break

    cam.release()
    cv2.destroyAllWindows()
    return ids_reais, ids_previstos

def criar_matriz_confusao(ids_reais, ids_previstos):
    cm = confusion_matrix(ids_reais, ids_previstos)
    labels = sorted(set(ids_reais))  # Classes reais detectadas durante o teste

    print("\nMatriz de Confusão:")
    print(cm)

    print("\nRelatório de Classificação:")
    print(classification_report(ids_reais, ids_previstos, labels=labels))

    acc = accuracy_score(ids_reais, ids_previstos)
    print(f"\nAcurácia: {acc:.2f}")

if __name__ == "__main__":
    ids_reais, ids_previstos = reconhecer_e_coletar_dados()
    if ids_reais and ids_previstos:
        criar_matriz_confusao(ids_reais, ids_previstos)
    else:
        print("[INFO] Nenhum dado coletado.")