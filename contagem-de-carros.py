import cv2
import numpy as np
import sys

VIDEO = "Dados/Ponte.mp4";
capture = cv2.VideoCapture(VIDEO)

algoritmo_types = ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2'];
algoritmoSelecionado = algoritmo_types[1];

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


background_substractor = Substractor(algoritmoSelecionado)

# Identificação de carros
w_min = 40  # Largura minima do retangulo
h_min = 40  # Altura minima do retangulo
offset = 2  # Offset para o retangulo - Margem de erro para pixel -> Maior offset mais conta a quantidade de carros
linha_ROI = 620  # Linha de ROI - Posição da linha de contagem de carros -  a partir dessa linha ele irá contar os carros
carros = 0


# y,x,w,h -> Posição do carro dentro do video
# Retorna o centro do carro ou objeto que deseja identificar
# Ao passar na linha de ROI apenas centroide vai contar quando ele passar pela linha
# A primeira parte da linha não vai contar pois ele não passou completamente
# Gera controle maior de contagem
def Centroide(x, y, w, h):
    """"
    :param x: Posição x do carro
    :param y: Posição y do carro
    :param w: Largura do carro
    :param h: Altura do carro
    :return: Retorna tupla que contém as coordenadas do centro do carro
    """
    x1 = w // 2
    y1 = h // 2
    cx = x + x1
    cy = y + y1
    return cx, cy


# Função para detecção do carro
detec = []


def set_info(detec):
    global carros
    for (x, y) in detec:
        # Se linha roi + offset for maior que y e y for maior que linha roi - offset
        if (linha_ROI + offset) > y > (linha_ROI - offset):
            carros += 1
            # Definindo a linha roi -> Desenha a linha de contagem de carros
            cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (0, 127, 255), 3)
            # Remove o carro da lista de detecção
            detec.remove((x, y))
            print("Carros detectados até o momento: " + str(carros))


def show_info(frame, mask):
    text = "Carros: " + str(carros)
    # Adiciona texto ao frame
    cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow('Frame', frame)
    cv2.imshow('Detectar', mask)


# Tipos de filtragens
def Kernel(KERNEL_TYPE):
    match KERNEL_TYPE:
        case 'dilation':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Matriz 3x3
        case 'opening':
            kernel = np.ones((3, 3), np.uint8)  # Matriz 3x3 de inteiros
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
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'),
                                    iterations=2)  # Faz filtro close duas vezes
        case 'combine':
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
            dilation = cv2.dilate(opening, Kernel('dilation'), iterations=2)
            return dilation  # Custo maior porém maior qualidade
        case _:
            return print("Filtro não foi feito"), sys.exit(1)


# Função principal - Aplica as mascaras e os filtros
while True:
    hasFrame, frame = capture.read()
    if not hasFrame:
        break

    mask = background_substractor.apply(frame)
    mask = Filter(mask, 'combine')

    # Encontrar contornos
    contorno, img = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Desenhar linha de ROI, linha de contagem de carros
    cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (255, 127, 0), 3)
    # Tudo dentro do contorno vai começão a ser desenhado e identificado
    for (i, c) in enumerate(contorno):
        # Parametros centroide, para fazer identificação do carro
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= w_min) and (h >= h_min)
        # Se não for um contorno válido ele segue o cód, caso seja uma pessoa ele não irá contar e seguir o cód normalmente.
        if not validar_contorno:
            continue
        # Faz retangulo ao redor do objeto de interesse
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Definir centroides
        centro = Centroide(x, y, w, h)
        # Adiciona o centroide a lista de detecção
        detec.append(centro)
        cv2.circle(frame, centro, 4, (0, 0, 255), -1)
    # Definir informações e mostrar na tela
    set_info(detec)
    show_info(frame, mask)

    # Mostrar por um tempo o frame
    if cv2.waitKey(1) == 27:  # ESC
        break

cv2.destroyAllWindows()
capture.release()
