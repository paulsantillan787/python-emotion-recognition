import os
import cv2
import mediapipe as mp
import numpy as np

# Inicializamos MediaPipe para detección de rostros
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Ruta de la carpeta de entrada y la carpeta donde se almacenarán los puntos faciales
input_folder = 'fotogramas'
output_folder = 'puntos-faciales'

# Función para procesar cada imagen
def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        return  # Si no se puede leer la imagen, saltamos el proceso
    
    # Convertimos la imagen a RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detección de puntos faciales
    results = face_mesh.process(rgb_image)
    
    # Verificamos si se detectaron puntos faciales
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = []
        
        # Extraemos los 468 puntos faciales
        for lm in face_landmarks.landmark:
            x, y, z = lm.x, lm.y, lm.z
            landmarks.append([x, y, z])
        
        # Guardamos los puntos faciales como un archivo .npy
        np.save(output_path, np.array(landmarks))

# Función para procesar todas las imágenes de la carpeta de entrada
def process_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.jpg'):
                # Construimos las rutas de entrada y salida
                relative_path = os.path.relpath(root, input_folder)
                input_image_path = os.path.join(root, file)
                output_dir = os.path.join(output_folder, relative_path)
                output_file_path = os.path.join(output_dir, file.replace('.jpg', '.npy'))
                
                # Creamos la carpeta de salida si no existe
                os.makedirs(output_dir, exist_ok=True)
                
                # Procesamos la imagen y guardamos los puntos faciales
                process_image(input_image_path, output_file_path)

# Ejecutamos el procesamiento
process_folder(input_folder, output_folder)

