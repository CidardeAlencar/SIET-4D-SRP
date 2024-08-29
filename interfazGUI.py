import tkinter as tk
from tkinter import font
from tkinter import PhotoImage
from PIL import Image, ImageTk
import subprocess

# Funciones para los botones
def posicion_de_pie():
    subprocess.Popen(["python", "posPie.py"])

def posicion_de_arrodillado():
    subprocess.Popen(["python", "posRod.py"])
    
def instruccion_posicion_de_pie():
    subprocess.Popen(["python", "posPieAntigua.py"])

def instruccion_posicion_de_arrodillado():
    subprocess.Popen(["python", "posRodAntigua.py"])

# Función para salir del modo pantalla completa
def salir_fullscreen(event):
    root.attributes('-fullscreen', False)

# Crear la ventana principal
root = tk.Tk()
root.title("SIET-4D")
root.attributes('-fullscreen', True)  # Pantalla completa
root.bind("<Escape>", salir_fullscreen)  # Vincular la tecla Esc para salir del modo pantalla completa

# Colores
bg_color = "#000000"  # Azul oscuro
fg_color = "#D9D9D8"  # Plomo
background_color = "#FFFACD"  # Amarillo pastel
green = "#0B600A"

# Configurar el fondo de la ventana
root.configure(bg=background_color)

# Crear una fuente grande para la etiqueta
label_font_title = font.Font(family="Helvetica", size=42, weight="bold")
label_font = font.Font(family="Helvetica", size=22, weight="bold")

# Crear un marco para centrar el contenido
frame = tk.Frame(root, bg=background_color)
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Crear una etiqueta grande
label = tk.Label(frame, text="Bienvenido al Sistema de Reconocimiento de Posiciones", 
                 font=label_font_title, fg=bg_color, bg=background_color, wraplength=900)
label.grid(row=0, column=0, columnspan=2, pady=20)

# Cargar y redimensionar la imagen usando PIL
image_path = "imagen1.png"
original_image = Image.open(image_path)
resized_image = original_image.resize((600, 500))  # Cambia el tamaño de la imagen (ancho, alto)
imagen = ImageTk.PhotoImage(resized_image)

# Crear un widget Label para mostrar la imagen
image_label = tk.Label(frame, image=imagen, bg=background_color)
image_label.grid(row=1, column=0, columnspan=2, pady=10)

# Crear los botones y organizarlos en una cuadrícula de 2x2
btn_pie = tk.Button(frame, text="Instrucción Posición de Pie", command=instruccion_posicion_de_pie, bg=green, fg=fg_color, font=label_font, width=25)
btn_pie.grid(row=2, column=0, padx=10, pady=10)

btn_arrodillado = tk.Button(frame, text="Instrucción Posición de Rodilla", command=instruccion_posicion_de_arrodillado, bg=green, fg=fg_color, font=label_font, width=25)
btn_arrodillado.grid(row=2, column=1, padx=10, pady=10)

btn_pie_eval = tk.Button(frame, text="Evaluación Posición de Pie", command=posicion_de_pie, bg=bg_color, fg=fg_color, font=label_font, width=25)
btn_pie_eval.grid(row=3, column=0, padx=10, pady=10)

btn_arrodillado_eval = tk.Button(frame, text="Evaluación Posición de Rodilla", command=posicion_de_arrodillado, bg=bg_color, fg=fg_color, font=label_font, width=25)
btn_arrodillado_eval.grid(row=3, column=1, padx=10, pady=10)

# Ejecutar el bucle principal
root.mainloop()
