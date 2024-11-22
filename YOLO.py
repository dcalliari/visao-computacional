import cv2


# conda install -c conda-forge ultralytics
from ultralytics import YOLO


VIDEO = "Dados/teste1.mp4"
capture = cv2.VideoCapture(VIDEO)

model = YOLO("yolo11n.pt")
ALVO_CLASSES = [2, 3, 5, 7]

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
    """ "
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


def processar_deteccoes(results, frame):
    """
    Processa as detecções do YOLO e retorna os centroides e o frame atualizado.
    """
    # Detecção do carro
    detec = []
    # Processar as detecções
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)  # Classe do objeto detectado
            if cls in ALVO_CLASSES:  # Verifica se a classe é de interesse
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0]
                )  # Coordenadas da caixa delimitadora
                w, h = x2 - x1, y2 - y1
                cx, cy = Centroide(x1, y1, w, h)  # Calcula o centroide
                detec.append((cx, cy))

                # Desenha a caixa e o centroide no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    return detec


def set_info(detec):
    global carros
    for x, y in detec:
        # Se linha roi + offset for maior que y e y for maior que linha roi - offset
        if (linha_ROI + offset) > y > (linha_ROI - offset):
            carros += 1
            # Definindo a linha roi -> Desenha a linha de contagem de carros
            cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (0, 127, 255), 3)
            # Remove o carro da lista de detecção
            detec.remove((x, y))
            print("Carros detectados até o momento: " + str(carros))


def show_info(frame, mask, congestion_status):
    # Adiciona texto ao frame
    text = "Carros: " + str(carros)
    if congestion_status:
        cv2.putText(
            frame,
            "Area congestionada",
            (250, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            5,
        )
    else:
        cv2.putText(
            frame, "Area livre", (450, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5
        )
    # Adiciona texto ao frame
    cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Frame", frame)
    cv2.imshow("Detectar", mask)


# Função para verificar congestionamento
def is_congested(vehicle_positions, threshold):
    """
    Verifica se a área está congestionada com base na densidade de veículos.

    :param vehicle_positions: Lista de posições dos veículos (centroides)
    :param threshold: Número mínimo de veículos para considerar congestionado
    :return: Verdadeiro se estiver congestionado, Falso caso contrário
    """
    return len(vehicle_positions) > threshold


# Função principal - Aplica as mascaras e os filtros
while True:
    hasFrame, frame = capture.read()

    if not hasFrame:
        break

    frame = frame[:, :-450]

    results = model(frame)

    detec = processar_deteccoes(results, frame)

    # Verificar se a área está congestionada
    congestion_status = is_congested(detec, threshold=15)

    # Definir informações e mostrar na tela
    set_info(detec)
    show_info(frame, carros, congestion_status)

    # Mostrar por um tempo o frame
    if cv2.waitKey(1) == 27:  # ESC
        break

cv2.destroyAllWindows()
capture.release()
paused = False


def toggle_pause(event, x, y, flags, param):
    global paused
    if event == cv2.EVENT_LBUTTONDOWN:
        paused = not paused


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", toggle_pause)

frame_skip = 5  # Number of frames to skip
frame_count = 0

while True:
    if not paused:
        hasFrame, frame = capture.read()
        if not hasFrame:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = frame[:, :-450]

        results = model(frame)

        detec = processar_deteccoes(results, frame)

        # Verificar se a área está congestionada
        congestion_status = is_congested(detec, threshold=17)

        # Definir informações e mostrar na tela
        set_info(detec)
        show_info(frame, carros, congestion_status)

    # Mostrar por um tempo o frame
    if cv2.waitKey(1) == 27:  # ESC
        break

cv2.destroyAllWindows()
capture.release()
