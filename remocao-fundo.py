import numpy as np;
import cv2;

VIDEO = "Dados/Rua.mp4";
# VIDEO = "Dados/Arco.mp4";

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

print(medianFrame);
# Ver imagem
cv2.imshow("Median Frame", medianFrame);
cv2.waitKey(0);

# Salvar imagem
# cv2.imwrite("Dados/imagem/MedianFrame.png", medianFrame);