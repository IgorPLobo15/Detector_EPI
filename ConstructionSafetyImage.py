from ultralytics import YOLO
import cv2
import math

# Carregar o modelo
model = YOLO("best.pt")

# Carregar a imagem
img = cv2.imread('Images/newobject.jpg')

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
              'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

myColor = (0, 0, 255)

# Loop de detecção
while True:
    results = model(img, stream=True)

    # Fazer uma cópia da imagem original para desenhar os retângulos e texto
    img_copy = img.copy()

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

                # Desenhar o texto e retângulo na imagem
                cv2.putText(img_copy, f'{classNames[cls]} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, myColor, 2, cv2.LINE_AA)
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), myColor, 3)

    # Mostrar a imagem com as detecções
    cv2.imshow("Image", img_copy)

    # Aguardar 1 ms para ver se uma tecla foi pressionada; se for a tecla 'q', sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar as janelas
cv2.destroyAllWindows()
