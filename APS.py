import cv2
import math
from ultralytics import YOLO
from tkinter import Tk, Button, filedialog, Label, Canvas, Frame
from PIL import Image, ImageTk
import threading
import albumentations as A
import numpy as np


# Função para realizar o tratamento de imagem
def preprocess_image(img):
    transforms = A.Compose([
        A.Blur(p=0.01, blur_limit=(3, 7)),
        A.MedianBlur(p=0.01, blur_limit=(3, 7)),
        A.ToGray(p=0.01, num_output_channels=3, method='weighted_average'),
        A.CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
    ])

    img_transformed = transforms(image=img)['image']
    return img_transformed


# Função para realizar a detecção de objetos
def run_yolo_detection(source, canvas, is_video=False, is_webcam=False):
    model = YOLO("best.pt")
    classNames = ['Escavadeira', 'Luvas', 'Capacete', 'Escada', 'Mascara', 'Sem Capacete', 'Sem Mascara',
                  'Sem Colete de Seguranca', 'Pessoa', 'SUV', 'Cone de Seguranca', 'Colete de Seguranca',
                  'Onibus', 'Caminhao Basculante', 'Hidrante', 'Maquinas', 'Van Pequena', 'Sedan', 'Caminhao Semi-Reboque',
                  'Reboque', 'Caminhao e Reboque', 'Caminhao', 'Van', 'Veiculo', 'Carregadeira']

    run_yolo_detection.running = True

    def process_frame(img):
        img = preprocess_image(img)  # Tratamento da imagem
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                # Definindo a cor com base na classe detectada
                if conf > 0.5:
                    if currentClass in ['Sem Capacete', 'Sem Colete de Seguranca', 'Sem Mascara']:
                        myColor = (0, 0, 255)  # Vermelho para ausência de EPI
                    elif currentClass in ['Capacete', 'Colete de Seguranca', 'Mascara']:
                        myColor = (0, 255, 0)  # Verde para presença de EPI
                    else:
                        myColor = (255, 0, 0)  # Azul para outros objetos

                    # Desenhar o retângulo e adicionar o texto
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                    img = cv2.putText(img, f'{currentClass}', (x1, y1),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img

    def display_frame(img):
        # Converter para formato suportado pelo Tkinter (RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Redimensionar imagem para caber no canvas e centralizar
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        img_width, img_height = img_pil.size

        # Calcular as proporções para centralizar
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        # Use Image.Resampling.LANCZOS no lugar de Image.ANTIALIAS
        img_pil_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil_resized)

        # Limpar canvas e centralizar a imagem
        canvas.delete("all")
        canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2, anchor="nw", image=img_tk)
        canvas.image = img_tk

    if hasattr(run_yolo_detection, "cap"):
        run_yolo_detection.cap.release()  # Libera a captura anterior

    if is_webcam or is_video:
        run_yolo_detection.cap = cv2.VideoCapture(0 if is_webcam else source)

        def process_video():
            while run_yolo_detection.cap.isOpened() and run_yolo_detection.running:
                ret, frame = run_yolo_detection.cap.read()
                if not ret:
                    break

                frame = process_frame(frame)
                display_frame(frame)
                canvas.update_idletasks()

            run_yolo_detection.cap.release()

        # Rodar o processamento de vídeo em thread separada para não travar a interface
        threading.Thread(target=process_video, daemon=True).start()

    else:
        img = cv2.imread(source)
        img = process_frame(img)
        display_frame(img)


# Função para parar a detecção de vídeo ou webcam
def stop_detection():
    run_yolo_detection.running = False
    if hasattr(run_yolo_detection, "cap"):
        run_yolo_detection.cap.release()


# Função para escolher arquivo de imagem
def open_image(canvas):
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        stop_detection()  # Para qualquer vídeo anterior
        run_yolo_detection(file_path, canvas, is_video=False)


# Função para escolher arquivo de vídeo
def open_video(canvas):
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if file_path:
        stop_detection()  # Para qualquer vídeo anterior
        run_yolo_detection(file_path, canvas, is_video=True)


# Função para capturar webcam
def open_webcam(canvas):
    stop_detection()  # Para qualquer vídeo anterior
    run_yolo_detection(0, canvas, is_video=True, is_webcam=True)


# Função para fechar a janela principal e liberar os recursos
def close_window():
    stop_detection()  # Libera a captura antes de fechar
    root.quit()
    root.destroy()


# Interface gráfica com Tkinter
root = Tk()
root.title("Detector de EPI")

# Título
title_label = Label(root, text="Escolha uma opção para detecção", font=("Helvetica", 16))
title_label.pack(pady=20)

# Frame para os botões
button_frame = Frame(root)
button_frame.pack(pady=10)

# Botão para carregar imagem
btn_image = Button(button_frame, text="Carregar Imagem", command=lambda: open_image(canvas), width=20, height=2)
btn_image.grid(row=0, column=0, padx=5)

# Botão para carregar vídeo
btn_video = Button(button_frame, text="Carregar Vídeo", command=lambda: open_video(canvas), width=20, height=2)
btn_video.grid(row=0, column=1, padx=5)

# Botão para capturar da Webcam
btn_webcam = Button(button_frame, text="Abrir Webcam", command=lambda: open_webcam(canvas), width=20, height=2)
btn_webcam.grid(row=0, column=2, padx=5)

# Botão para parar detecção
btn_stop = Button(button_frame, text="Parar Detecção", command=stop_detection, width=20, height=2, bg="yellow")
btn_stop.grid(row=1, column=1, pady=10)

# Botão de sair
btn_exit = Button(root, text="Sair", command=close_window, width=30, height=2, bg="red", fg="white")
btn_exit.pack(pady=20)

# Canvas para exibir a imagem processada
canvas = Canvas(root, width=800, height=600)
canvas.pack(pady=20)

# Tornar a interface responsiva ao redimensionamento da janela
def resize_canvas(event):
    canvas.config(width=event.width, height=event.height)

canvas.bind("<Configure>", resize_canvas)

root.geometry("900x800")
root.mainloop()
