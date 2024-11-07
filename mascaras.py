# Tipos de mascaras
# KNN, GMG,CNT,MOG,MOG2
# KNN -> Agrupa grupos de acordo com suas semelhanças
# GMG -> Teorema de Bayes
# CNT -> É utilizado na contagem de frames para encontrar o fundo ou o objeto de primeiro plano da imagem.
# MOG -> MOG é uma abreviação de Mixture of Gaussians, que em português podemos adaptar para Mistura de fundo adaptativa.
# Nela é feita uma distribuição gaussiana (também conhecida como distribuição normal para cada pixel, de forma que seja caracterizado por sua intensidade no espaço de cores RGB.

import cv2;
import sys;

from dill.temp import capture
from sympy.strategies.core import switch

VIDEO = "Dados/Ponte.mp4";

algoritmo = ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2'];
algoritmoSelecionado = algoritmo[3];


def Substractor(algoritmoSelecionado):
    match algoritmoSelecionado:
        case 'KNN':
            return cv2.createBackgroundSubtractorKNN();
        case 'GMG':
            return cv2.createBackgroundSubtractorGMG();
        case 'CNT':
            return cv2.bgsegm.createBackgroundSubtractorCNT();
        case 'MOG':
            return cv2.bgsegm.createBackgroundSubtractorMOG();
        case 'MOG2':
            return cv2.createBackgroundSubtractorMOG2();
        case _:
            return print("Algoritmo não encontrado"), sys.exit(1);


capture = cv2.VideoCapture(VIDEO);
subtractor = Substractor(algoritmoSelecionado);


def main():
    while (capture.isOpened()):
        hasFrame, frame = capture.read();

        if not hasFrame:
            print("Fim dos Frames");
            break;

        mask = subtractor.apply(frame);

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow("Frame", frame);
        cv2.imshow("Mask", mask);

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break;

main();