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

from mascaras import background_substractor

VIDEO = "Dados/Ponte.mp4";

algoritmo_types = ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2'];
algoritmoSelecionado = algoritmo_types[4];

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
print("Dilation: ")
print(Kernel('dilation'))
print("Opening: ")
print(Kernel('opening'))
print("Closing: ")
print(Kernel('closing'))

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

def main():
    while (capture.isOpened()):
        hasFrame, frame = capture.read();

        if not hasFrame:
            print("Fim dos Frames");
            break;

        frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)
        mask = background_substractor.apply(frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break;

        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)


main();
