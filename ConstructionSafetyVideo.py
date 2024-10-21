from ultralytics import YOLO
import cv2
import math

# Carregar o modelo
model = YOLO("best.pt")

# Inicializar a webcam (pode ser 0, 1 ou outro índice dependendo da sua webcam)
cap = cv2.VideoCapture(0)

# Nomes das classes de objetos
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
              'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

myColor = (0, 0, 255)

while True:
    # Capturar o frame da webcam
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar o vídeo")
        break

    # Passar o frame para o modelo YOLO
    results = model(frame, stream=True)

    # Fazer uma cópia do frame original para desenhar as detecções
    frame_copy = frame.copy()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordenadas da Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confiança
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Nome da classe
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)

            if conf > 0.5:
                # Definir a cor de acordo com a classe detectada
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0, 255)  # Vermelho para itens em falta
                elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                    myColor = (0, 255, 0)  # Verde para itens corretos
                else:
                    myColor = (255, 0, 0)  # Azul para outros objetos

                # Desenhar o texto e o retângulo no frame
                cv2.putText(frame_copy, f'{classNames[cls]} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, myColor, 2, cv2.LINE_AA)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), myColor, 3)

    # Mostrar o frame com as detecções
    cv2.imshow("Webcam", frame_copy)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura da webcam e fechar as janelas
cap.release()
cv2.destroyAllWindows()
