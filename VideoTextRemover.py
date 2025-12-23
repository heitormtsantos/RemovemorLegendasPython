import cv2
import os
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  # para mostrar o frame no Tkinter

# ---------- FUNÇÃO PRINCIPAL DE PROCESSAMENTO ---------- #

def process_video(video_path, band_top_frac=0.55, band_bottom_frac=0.95,
                  band_left_frac=0.0, band_right_frac=1.0,
                  thresh_val=230, min_pixels_text=150, clean_weight=0.75,
                  dilation_iter=10, use_edges=False):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    band_top_frac = max(0.0, min(1.0, band_top_frac))
    band_bottom_frac = max(0.0, min(1.0, band_bottom_frac))
    if band_bottom_frac <= band_top_frac:
        raise ValueError("O fim da faixa (Y) deve ser maior que o início.")
    
    band_left_frac = max(0.0, min(1.0, band_left_frac))
    band_right_frac = max(0.0, min(1.0, band_right_frac))
    if band_right_frac <= band_left_frac:
        raise ValueError("O fim da faixa (X) deve ser maior que o início.")

    clean_weight = max(0.0, min(1.0, clean_weight))

    # --- NOME DO ARQUIVO E PASTA DE SAÍDA ---
    video_name = os.path.basename(video_path)
    name_no_ext, _ = os.path.splitext(video_name)

    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    output_folder = os.path.join(downloads, "legenda_removida")
    os.makedirs(output_folder, exist_ok=True)

    temp_output = os.path.join(output_folder, f"{name_no_ext}_temp.avi")
    final_output = os.path.join(output_folder, f"{name_no_ext}_sem_legenda.mp4")

    start_time = time.time()

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo.")

    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = vid.get(cv2.CAP_PROP_FPS) or 25
    width       = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processando: {video_name}")
    print(f"Frames: {frame_count}, FPS: {fps}, Resolução: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    if not writer.isOpened():
        vid.release()
        raise RuntimeError("Não foi possível criar AVI temporário.")

    band_top = int(height * band_top_frac)
    band_bottom = int(height * band_bottom_frac)
    band_left = int(width * band_left_frac)
    band_right = int(width * band_right_frac)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    CLEAN_WEIGHT = clean_weight
    ORIG_WEIGHT  = 1.0 - CLEAN_WEIGHT

    frame_idx = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        roi = frame[band_top:band_bottom, band_left:band_right]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        _, bin_roi = cv2.threshold(gray_roi, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Se ativado, usa detecção de bordas (Canny) para reforçar a máscara
        # Isso ajuda a capturar legendas claras ou com bordas definidas que o threshold ignora
        if use_edges:
            # Canny com limiares conservadores para pegar texto
            edges = cv2.Canny(gray_roi, 50, 150)
            bin_roi = cv2.bitwise_or(bin_roi, edges)

        bin_roi = cv2.morphologyEx(bin_roi, cv2.MORPH_CLOSE, kernel)

        # Dilatação configurável
        if dilation_iter > 0:
            bin_roi = cv2.dilate(bin_roi, kernel, iterations=dilation_iter)

        white_pixels = cv2.countNonZero(bin_roi)

        if white_pixels > min_pixels_text:
            # Inpainting direto com a máscara binária dilatada
            # Voltamos a processar a ROI inteira para evitar artefatos de recorte (faixas)
            cleaned_roi = cv2.inpaint(roi, bin_roi, 3, cv2.INPAINT_TELEA)
            
            # Se a densidade for 100%, não misturamos com o original para evitar fantasmas
            if CLEAN_WEIGHT >= 1.0:
                frame[band_top:band_bottom, band_left:band_right] = cleaned_roi
            else:
                blended_roi = cv2.addWeighted(cleaned_roi, CLEAN_WEIGHT, roi, ORIG_WEIGHT, 0)
                frame[band_top:band_bottom, band_left:band_right] = blended_roi

        writer.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processado {frame_idx}/{frame_count} frames...")

    vid.release()
    writer.release()

    exec_time = time.time() - start_time

    # ---------- CONVERSÃO PARA H.264 COM ÁUDIO ORIGINAL ---------- #
    print("\nConvertendo para MP4 H.264 e copiando áudio original...")

    # -i temp_output: vídeo processado (sem áudio)
    # -i video_path: vídeo original (fonte do áudio)
    # -c:v libx264: recodifica o vídeo
    # -c:a copy: copia o áudio original sem reprocessar (rápido e sem perda)
    # -map 0:v:0: pega o vídeo da primeira entrada (temp_output)
    # -map 1:a:0: pega o áudio da segunda entrada (video_path)
    # -shortest: garante que pare quando o menor stream acabar (evita loops)
    
    ffmpeg_cmd = (
        f'ffmpeg -y -i "{temp_output}" -i "{video_path}" '
        f'-map 0:v:0 -map 1:a:0? '  # O '?' evita erro se o vídeo original não tiver áudio
        f'-c:v libx264 -pix_fmt yuv420p -preset veryfast '
        f'-c:a copy -shortest '
        f'"{final_output}"'
    )

    os.system(ffmpeg_cmd)
    os.remove(temp_output)

    print(f"\nVídeo final salvo em: {final_output}")
    print(f"Tempo total: {exec_time:.2f} segundos")

    return final_output, exec_time


# ---------- INTERFACE GRÁFICA (TKINTER) ---------- #

selected_video_path = None
preview_frame = None
preview_img_tk = None
rect_id = None

MAX_PREVIEW_W = 380
MAX_PREVIEW_H = 450

def choose_video():
    global selected_video_path, preview_frame, preview_img_tk, rect_id

    file_path = filedialog.askopenfilename(
        title="Escolher vídeo",
        filetypes=(
            ("Arquivos de vídeo", "*.mp4;*.avi;*.mkv;*.mov"),
            ("Todos os arquivos", "*.*")
        )
    )
    if not file_path:
        return

    selected_video_path = file_path
    label_video.config(text=f"Vídeo selecionado:\n{file_path}")

    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Erro", "Não foi possível ler o primeiro frame do vídeo.")
        return

    preview_frame = frame

    h, w = frame.shape[:2]
    scale = min(MAX_PREVIEW_W / float(w), MAX_PREVIEW_H / float(h))
    new_w = int(w * scale)
    new_h = int(h * scale)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
    preview_img_tk = ImageTk.PhotoImage(img_pil)

    canvas_preview.config(width=new_w, height=new_h)
    canvas_preview.delete("all")
    canvas_preview.create_image(0, 0, anchor="nw", image=preview_img_tk)

    rect_id = None
    draw_band_rectangle()

def draw_band_rectangle(*args):
    global rect_id, preview_frame, preview_img_tk

    if preview_frame is None or preview_img_tk is None:
        return

    canvas_w = canvas_preview.winfo_width()
    canvas_h = canvas_preview.winfo_height()

    top_frac = band_top_var.get() / 100.0
    bottom_frac = band_bottom_var.get() / 100.0
    left_frac = band_left_var.get() / 100.0
    right_frac = band_right_var.get() / 100.0

    if bottom_frac - top_frac < 0.02:
        bottom_frac = top_frac + 0.02
        band_bottom_var.set(bottom_frac * 100)

    if right_frac - left_frac < 0.02:
        right_frac = left_frac + 0.02
        band_right_var.set(right_frac * 100)

    y1 = int(canvas_h * top_frac)
    y2 = int(canvas_h * bottom_frac)
    x1 = int(canvas_w * left_frac)
    x2 = int(canvas_w * right_frac)

    if rect_id is None:
        rect_id = canvas_preview.create_rectangle(
            x1, y1, x2, y2,
            outline="red", width=2
        )
    else:
        canvas_preview.coords(rect_id, x1, y1, x2, y2)

def run_processing():
    global selected_video_path
    if not selected_video_path:
        messagebox.showwarning("Atenção", "Escolha um vídeo primeiro.")
        return

    top_frac = band_top_var.get() / 100.0
    bottom_frac = band_bottom_var.get() / 100.0
    left_frac = band_left_var.get() / 100.0
    right_frac = band_right_var.get() / 100.0
    
    thresh = int(threshold_var.get())
    dilation = int(dilation_var.get())
    use_canny = edges_var.get()

    if bottom_frac <= top_frac:
        messagebox.showerror("Erro", "O fim da faixa (Y) deve ser maior que o início.")
        return
    
    if right_frac <= left_frac:
        messagebox.showerror("Erro", "O fim da faixa (X) deve ser maior que o início.")
        return

    density = density_var.get() / 100.0

    btn_run.config(state="disabled")
    root.update_idletasks()

    try:
        messagebox.showinfo(
            "Processando",
            "Iniciando remoção das legendas...\nIsso pode demorar alguns minutos."
        )
        output_path, exec_time = process_video(
            selected_video_path,
            band_top_frac=top_frac,
            band_bottom_frac=bottom_frac,
            band_left_frac=left_frac,
            band_right_frac=right_frac,
            thresh_val=thresh,
            min_pixels_text=150,
            clean_weight=density,
            dilation_iter=dilation,
            use_edges=use_canny
        )
        messagebox.showinfo(
            "Concluído",
            f"Vídeo processado com sucesso!\n\n"
            f"Saída: {output_path}\n"
            f"Tempo: {exec_time:.2f} s"
        )
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro:\n{e}")
    finally:
        btn_run.config(state="normal")


# ---------- CRIA JANELA ---------- #

root = tk.Tk()
root.title("Removedor de Legendas 9:16 (Inpainting)")

root.geometry("1000x750")
root.resizable(True, True)

title_label = tk.Label(root, text="Removedor de Legendas 9:16", font=("Arial", 16, "bold"))
title_label.pack(pady=5)

frame_top = tk.Frame(root)
frame_top.pack(pady=5)

btn_choose = tk.Button(frame_top, text="Escolher vídeo...", command=choose_video, width=20)
btn_choose.grid(row=0, column=0, padx=10)

btn_run = tk.Button(frame_top, text="Remover legenda", command=run_processing, width=20)
btn_run.grid(row=0, column=1, padx=10)

label_video = tk.Label(root, text="Nenhum vídeo selecionado.", wraplength=860, justify="center")
label_video.pack(pady=5)

frame_middle = tk.Frame(root)
frame_middle.pack(pady=5, fill="x")

canvas_preview = tk.Canvas(frame_middle, width=MAX_PREVIEW_W, height=MAX_PREVIEW_H, bg="black")
canvas_preview.pack(side="left", padx=10, pady=5)

frame_controls = tk.Frame(frame_middle)
frame_controls.pack(side="left", padx=20, pady=5, fill="y")

band_top_var = tk.DoubleVar(value=55.0)
band_bottom_var = tk.DoubleVar(value=95.0)
band_left_var = tk.DoubleVar(value=0.0)
band_right_var = tk.DoubleVar(value=100.0)

threshold_var = tk.DoubleVar(value=230.0)
dilation_var = tk.DoubleVar(value=10.0)
density_var = tk.DoubleVar(value=75.0)
edges_var = tk.BooleanVar(value=False)

# --- Sliders de Área ---
label_area = tk.Label(frame_controls, text="Área de Remoção", font=("Arial", 10, "bold"))
label_area.pack(pady=(0, 5))

frame_sliders_area = tk.Frame(frame_controls)
frame_sliders_area.pack(fill="x")

# Top
tk.Label(frame_sliders_area, text="Topo Y%").pack(anchor="w")
slider_top = tk.Scale(frame_sliders_area, from_=0, to=100,
                      orient="horizontal", variable=band_top_var,
                      command=draw_band_rectangle)
slider_top.pack(fill="x")

# Bottom
tk.Label(frame_sliders_area, text="Base Y%").pack(anchor="w")
slider_bottom = tk.Scale(frame_sliders_area, from_=0, to=100,
                         orient="horizontal", variable=band_bottom_var,
                         command=draw_band_rectangle)
slider_bottom.pack(fill="x")

# Left
tk.Label(frame_sliders_area, text="Esq X%").pack(anchor="w")
slider_left = tk.Scale(frame_sliders_area, from_=0, to=100,
                      orient="horizontal", variable=band_left_var,
                      command=draw_band_rectangle)
slider_left.pack(fill="x")

# Right
tk.Label(frame_sliders_area, text="Dir X%").pack(anchor="w")
slider_right = tk.Scale(frame_sliders_area, from_=0, to=100,
                         orient="horizontal", variable=band_right_var,
                         command=draw_band_rectangle)
slider_right.pack(fill="x")

# --- Sliders de Ajuste ---
label_adjust = tk.Label(frame_controls, text="Ajustes Finos", font=("Arial", 10, "bold"))
label_adjust.pack(pady=(15, 5))

# Threshold
tk.Label(frame_controls, text="Limiar de Brilho (0-255)").pack(anchor="w")
slider_thresh = tk.Scale(frame_controls, from_=0, to=255,
                         orient="horizontal", variable=threshold_var)
slider_thresh.pack(fill="x")

# Dilatação
tk.Label(frame_controls, text="Espessura Máscara").pack(anchor="w")
slider_dilation = tk.Scale(frame_controls, from_=0, to=50,
                           orient="horizontal", variable=dilation_var)
slider_dilation.pack(fill="x")

# Reforço de Bordas
check_edges = tk.Checkbutton(frame_controls, text="Reforçar Bordas (Canny)", variable=edges_var)
check_edges.pack(pady=5, anchor="w")

# Densidade
tk.Label(frame_controls, text="Opacidade Remoção (%)").pack(anchor="w")
slider_density = tk.Scale(frame_controls, from_=0, to=100,
                          orient="horizontal", variable=density_var)
slider_density.pack(fill="x")

info_label = tk.Label(
    root,
    text=("Ajuste a área da legenda e a densidade do apagamento.\n"
          "O vídeo final será salvo em MP4 H.264 (compatível com Chrome)."),
    font=("Arial", 9),
    justify="center"
)
info_label.pack(pady=5)

root.mainloop()
