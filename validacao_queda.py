import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def detectar_e_coletar_quedas():
    # Inicializar o detector de pose
    detector = PoseDetector()
    ids_reais = []  # IDs reais: 1 para queda, 0 para sem queda
    ids_previstos = []  # IDs previstos pelo modelo

    # Configuração da câmera
    cam = cv2.VideoCapture(1)  # Alterar para o índice correto da câmera, se necessário
    cam.set(3, 1920)  # Largura
    cam.set(4, 1080)  # Altura

    print("[INFO] Use as teclas (0 para 'Sem Queda', 1 para 'Queda') ou ESC para sair.")
    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERRO] Não foi possível acessar a câmera.")
            break

        # Detectar pose
        img = detector.findPose(img, draw=False)
        pontos, bbox = detector.findPosition(img, draw=False)

        # Determinar se é uma queda
        queda_detectada = False
        if pontos:
            cabeca = pontos[0][1]  # Coordenada Y da cabeça
            joelho = pontos[26][1] if len(pontos) > 26 else 0  # Coordenada Y do joelho
            queda_detectada = joelho - cabeca <= 0  # Lógica para detectar queda

        # Determinar rótulos de queda
        queda_prevista_texto = "Queda" if queda_detectada else "Sem Queda"

        # Exibir informações na tela
        if queda_detectada:
            cv2.putText(img, "QUEDA DETECTADA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "SEM QUEDA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Exibir IDs reais e previstos
        if ids_reais:
            ultima_real = ids_reais[-1]
            ultima_prevista = ids_previstos[-1] if ids_previstos else "N/A"
            cv2.putText(img, f"Queda Real: {ultima_real}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Queda Prevista: {ultima_prevista}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Exibir vídeo
        cv2.imshow('Deteccao de Quedas', img)

        # Obter tecla pressionada
        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # Pressione ESC para sair
            print("[INFO] Encerrando o programa.")
            break
        elif key == ord('0'):  # Tecla 0 para "Sem Queda"
            ids_reais.append(0)
            ids_previstos.append(1 if queda_detectada else 0)
            print(f"[INFO] Entrada: Sem Queda (Real: 0, Previsto: {1 if queda_detectada else 0})")
        elif key == ord('1'):  # Tecla 1 para "Queda"
            ids_reais.append(1)
            ids_previstos.append(1 if queda_detectada else 0)
            print(f"[INFO] Entrada: Queda (Real: 1, Previsto: {1 if queda_detectada else 0})")

        if len(ids_reais) >= 100:  # Limitar número de amostras
            print("[INFO] Número máximo de amostras atingido. Encerrando.")
            break

    cam.release()
    cv2.destroyAllWindows()
    return ids_reais, ids_previstos

def criar_matriz_confusao(ids_reais, ids_previstos):
    # Gerar matriz de confusão
    cm = confusion_matrix(ids_reais, ids_previstos)
    labels = [0, 1]  # Classes: 0 (Sem Queda), 1 (Queda)

    # Exibir matriz de confusão
    print("\nMatriz de Confusão:")
    print(cm)

    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(ids_reais, ids_previstos, labels=labels))

    # Calcular e exibir a acurácia
    acc = accuracy_score(ids_reais, ids_previstos)
    print(f"\nAcurácia: {acc:.2f}")

if __name__ == "__main__":
    # Executar o detector de quedas
    ids_reais, ids_previstos = detectar_e_coletar_quedas()

    # Gerar a matriz de confusão
    if ids_reais and ids_previstos:
        criar_matriz_confusao(ids_reais, ids_previstos)
    else:
        print("[INFO] Nenhum dado coletado.")
