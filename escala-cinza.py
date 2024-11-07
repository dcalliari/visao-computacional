import numpy as np;
import cv2;
from time import sleep;

VIDEO = "Dados/Rua.mp4";
delay = 10;

capture = cv2.VideoCapture(VIDEO);
hasFrame, frame = capture.read();

# Pega frames aleatorios dos frames dos videos
# Pegando 72 frames aleatorios
framesIds = capture.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=75);

frames = [];

# Processa os frames, pegando dados dos frames e verificando se tem frame e adiciona a lista de frames

for fId in framesIds:
    capture.set(cv2.CAP_PROP_POS_FRAMES, fId);
    hasFrame, frame = capture.read();
    frames.append(frame);

# Faz a mediana dos frames para remover o fundo
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8);

# --- Escala Cinza ---
capture.set(cv2.CAP_PROP_POS_FRAMES, 0);
# Transformar frame em escala cinza
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY);
# Salva a imagem
cv2.imwrite("Dados/imagem/MedianFrameGray.png", grayMedianFrame);
# cv2.imshow("Gray Median Frame", grayMedianFrame);
# cv2.waitKey(0);

while (True):

    tempo=float(1/delay);
    sleep(tempo);

    hasFrame, frame = capture.read();

    if not hasFrame:
        print("Fim dos Frames");
        break;

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);

    # Equação para diferenciar os objetos do fundo -> saber o que é um carro ou pedestre etc
    # Subtração do que ta diferente deles
    dframe = cv2.absdiff(frameGray, grayMedianFrame);

    # Threshold -> é uma das técnicas de segmentação mais comuns em visão computacional e nos permite separar o primeiro plano
    # (ou seja, os objetos nos quais estamos interessados) do plano de fundo da imagem. Com essa separação,
    # podemos encontrar os possíveis objetos que se movimentam pelo vídeo.

    # THRESH_BINARY -> Carros brancos e fundo preto
    # THEREH_OTSU -> Faz a binarização automatica
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU);

    cv2.imshow("Diff", dframe);
    # cv2.imshow("Frame", frameGray);
# Fechar quando o video acaba
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break;

capture.release();






