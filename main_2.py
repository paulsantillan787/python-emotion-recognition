import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Cargar el modelo
model = load_model('./Modelos/model_estable_inestable_478_landmarks.h5')

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Iniciar la captura de video
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar los puntos faciales
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Extraer los puntos faciales
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks).flatten().reshape(1, -1)  # Aplanar y reestructurar

            # Hacer la predicción
            prediction = model.predict(landmarks)
            predicted_class = np.argmax(prediction, axis=1)

            # Mostrar el resultado de la predicción
            result_text = "Estable" if predicted_class[0] == 0 else "Inestable"
            cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen con los puntos faciales y la predicción
    cv2.imshow('Camara', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
