import cv2
import math
from ultralytics import YOLO
from tkinter import Tk, Button, filedialog, Label, Canvas, Frame, Text, Scrollbar, RIGHT, Y, END
from PIL import Image, ImageTk
import threading
import albumentations as A
import numpy as np

# Constantes
EPI_MODEL_PATH = "best.pt"
MATERIAL_MODEL_PATH = "materials_model.pt"
EPI_CLASS_NAMES = [
    'Escavadeira', 'Luvas', 'Capacete', 'Escada', 'Mascara', 'Sem Capacete',
    'Sem Mascara', 'Sem Colete de Seguranca', 'Pessoa', 'SUV', 'Cone de Seguranca',
    'Colete de Seguranca', 'Onibus', 'Caminhao Basculante', 'Hidrante', 'Maquinas',
    'Van Pequena', 'Sedan', 'Caminhao Semi-Reboque', 'Reboque', 'Caminhao e Reboque',
    'Caminhao', 'Van', 'Veiculo', 'Carregadeira'
]
MATERIAL_CLASS_NAMES = [
    'Escavadeira', 'material_tecido', 'material_plastico', 'Escada', 'material_tecido',
    'Sem Capacete', 'Sem Máscara', 'camisa_de_algodao', 'Pessoa', 'SUV', 'Cone de Segurança',
    'material_poliester', 'Ônibus', 'Caminhão Basculante', 'Hidrante', 'Máquinas',
    'Van Pequena', 'Sedan', 'Caminhão Semi-Reboque', 'Reboque', 'Caminhão e Reboque',
    'Caminhão', 'Van', 'Veículo', 'Carregadeira'
]

# Função para realizar o tratamento de imagem
def preprocess_image(img):
    """Aplica transformações na imagem para melhorar a detecção."""
    transforms = A.Compose([
        A.Blur(p=0.01, blur_limit=(3, 7)),
        A.MedianBlur(p=0.01, blur_limit=(3, 7)),
        A.ToGray(p=0.01, num_output_channels=3, method='weighted_average'),
        A.CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
    ])
    return transforms(image=img)['image']

# Função para realizar a detecção de objetos
def run_yolo_detection(source, canvas, report_text, is_video=False, is_webcam=False):
    """Executa a detecção de objetos usando os modelos YOLO."""
    epi_model = YOLO(EPI_MODEL_PATH)
    material_model = YOLO(MATERIAL_MODEL_PATH)
    run_yolo_detection.running = True

    def process_frame(img):
        """Processa cada frame para detecção de objetos."""
        img = preprocess_image(img)
        epi_results = epi_model(img, stream=True)
        report_text.delete(1.0, END)  # Limpa o relatório anterior
        for r in epi_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = EPI_CLASS_NAMES[cls]

                if conf > 0.5:
                    my_color = (0, 0, 255) if current_class in ['Sem Capacete', 'Sem Colete de Seguranca', 'Sem Mascara'] else (0, 255, 0) if current_class in ['Capacete', 'Colete de Seguranca', 'Mascara'] else (255, 0, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), my_color, 3)
                    cv2.putText(img, f'{current_class}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Recorte da região do EPI para análise de material
                    epi_region = img[y1:y2, x1:x2]
                    material_results = material_model(epi_region, stream=True)
                    material_counts = {}
                    for mr in material_results:
                        for mbox in mr.boxes:
                            mcls = int(mbox.cls[0])
                            material_class = MATERIAL_CLASS_NAMES[mcls]
                            material_counts[material_class] = material_counts.get(material_class, 0) + 1

                    # Atualiza o relatório de materiais
                    total_detections = sum(material_counts.values())
                    report_text.insert(END, f"Classe: {current_class}\n")
                    for material, count in material_counts.items():
                        percentage = (count / total_detections) * 100
                        report_text.insert(END, f"{material}: {percentage:.2f}%\n")
                    report_text.insert(END, "\n")

        return img

    def display_frame(img):
        """Exibe o frame processado no canvas."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
        img_width, img_height = img_pil.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width, new_height = int(img_width * ratio), int(img_height * ratio)
        img_pil_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil_resized)
        canvas.delete("all")
        canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2, anchor="nw", image=img_tk)
        canvas.image = img_tk

    if hasattr(run_yolo_detection, "cap"):
        run_yolo_detection.cap.release()

    if is_webcam or is_video:
        run_yolo_detection.cap = cv2.VideoCapture(0 if is_webcam else source)

        def process_video():
            """Processa o vídeo frame a frame."""
            while run_yolo_detection.cap.isOpened() and run_yolo_detection.running:
                ret, frame = run_yolo_detection.cap.read()
                if not ret:
                    break
                frame = process_frame(frame)
                display_frame(frame)
                canvas.update_idletasks()
            run_yolo_detection.cap.release()

        threading.Thread(target=process_video, daemon=True).start()
    else:
        img = cv2.imread(source)
        img = process_frame(img)
        display_frame(img)

def stop_detection():
    """Para a detecção de vídeo ou webcam."""
    run_yolo_detection.running = False
    if hasattr(run_yolo_detection, "cap"):
        run_yolo_detection.cap.release()

def open_image(canvas, report_text):
    """Abre um arquivo de imagem para detecção."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        stop_detection()
        run_yolo_detection(file_path, canvas, report_text, is_video=False)

def open_video(canvas, report_text):
    """Abre um arquivo de vídeo para detecção."""
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if file_path:
        stop_detection()
        run_yolo_detection(file_path, canvas, report_text, is_video=True)

def open_webcam(canvas, report_text):
    """Inicia a captura da webcam para detecção."""
    stop_detection()
    run_yolo_detection(0, canvas, report_text, is_video=True, is_webcam=True)

def close_window():
    """Fecha a janela principal e libera os recursos."""
    stop_detection()
    root.quit()
    root.destroy()

# Interface gráfica com Tkinter
root = Tk()
root.title("Detector de EPI e Material")

# Título
title_label = Label(root, text="Escolha uma opção para detecção", font=("Helvetica", 16))
title_label.pack(pady=20)

# Frame para os botões
button_frame = Frame(root)
button_frame.pack(pady=10)

# Botão para carregar imagem
btn_image = Button(button_frame, text="Carregar Imagem", command=lambda: open_image(canvas, report_text), width=20, height=2)
btn_image.grid(row=0, column=0, padx=5)

# Botão para carregar vídeo
btn_video = Button(button_frame, text="Carregar Vídeo", command=lambda: open_video(canvas, report_text), width=20, height=2)
btn_video.grid(row=0, column=1, padx=5)

# Botão para capturar da Webcam
btn_webcam = Button(button_frame, text="Abrir Webcam", command=lambda: open_webcam(canvas, report_text), width=20, height=2)
btn_webcam.grid(row=0, column=2, padx=5)

# Botão para parar detecção
btn_stop = Button(button_frame, text="Parar Detecção", command=stop_detection, width=20, height=2, bg="yellow")
btn_stop.grid(row=1, column=1, pady=10)

# Botão de sair
btn_exit = Button(root, text="Sair", command=close_window, width=30, height=2, bg="red", fg="white")
btn_exit.pack(pady=20)

# Frame para a exibição de imagem e relatório
display_frame = Frame(root)
display_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Canvas para exibir a imagem processada
canvas = Canvas(display_frame, width=800, height=600)
canvas.pack(side="left", fill="both", expand=True)

# Text widget para exibir o relatório de materiais
report_text = Text(display_frame, width=40, height=35)
report_text.pack(side="right", fill="both", expand=True)

# Tornar a interface responsiva ao redimensionamento da janela
def resize_canvas(event):
    canvas.config(width=event.width, height=event.height)

canvas.bind("<Configure>", resize_canvas)

root.geometry("1200x800")
root.mainloop()
