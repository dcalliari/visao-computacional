# Kernel
# Matriz usada para reduzir o tamanho de frames
# Também pode ser usado no desfoque, nitidez e iluminação de imagens por exemplo
# https://setosa.io/ev/image-kernels/

# Erosão - diminuir quantidade de pixels brancos
# Dilatação - aumentar quantidade de pixels brancos
# Abertura - remove ruídos da parte de fora
# Fechamento - remove ruídos da parte de dentro

# Kernel - numpy ou opencv

import cv2
import numpy as np
import sys

VIDEO = "Dados/Ponte.mp4";

algoritmo_types = ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2'];
algoritmoSelecionado = algoritmo_types[4];

# Tipos de filtragens
def Kernel(KERNEL_TYPE):
    match KERNEL_TYPE:
        case 'dilation':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # Matriz 3x3
        case 'opening':
            kernel = np.ones((3, 3), np.uint8) # Matriz 3x3 de inteiros
        case 'closing':
            kernel = np.ones((3, 3), np.uint8)
        case _:
            return print("Kernel não encontrado"), sys.exit(1);
    return kernel

# Adicionar filtragem aos videos usando openCV
def Filter(img, filter):
    match filter:
        case 'dilation':
            return cv2.dilate(img, Kernel('dilation'), iterations=2)
        case 'opening':
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
        case 'closing':
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2) # Faz filtro close duas vezes
        case 'combine':
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
            dilation = cv2.dilate(opening, Kernel('dilation'), iterations=2)
            return dilation # Custo maior porém maior qualidade
        case _:
            return print("Filtro não foi feito"), sys.exit(1)

# Adiciona a mascara ao frame
def Substractor(algoritmoSelecionado):
    match algoritmoSelecionado:
        case 'KNN':
            return cv2.createBackgroundSubtractorKNN();
        case 'GMG':
            return cv2.bgsegm.createBackgroundSubtractorGMG();
        case 'CNT':
            return cv2.bgsegm.createBackgroundSubtractorCNT();
        case 'MOG':
            return cv2.bgsegm.createBackgroundSubtractorMOG();
        case 'MOG2':
            return cv2.createBackgroundSubtractorMOG2();
        case _:
            return print("Algoritmo não encontrado"), sys.exit(1);

capture = cv2.VideoCapture(VIDEO)
background_substractor = Substractor(algoritmoSelecionado)

# Função principal - Aplica as mascaras e os filtros
def main():
    while (capture.isOpened()):
        hasFrame, frame = capture.read();

        if not hasFrame:
            print("Fim dos Frames");
            break;

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        mask = background_substractor.apply(frame)
        # Escolher filtro para aplicar na mascara
        mask_Filter  = Filter(mask, 'combine')
        # Aplicar mascara ao frame
        cars_after_mask = cv2.bitwise_and(frame, frame, mask=mask_Filter)

        cv2.imshow('Mask', cars_after_mask)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break;

main();
