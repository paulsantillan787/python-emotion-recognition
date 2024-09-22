import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model('emotion_model.h5')

# Inicializar dlib y el detector de rostros
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Asegúrate de tener este archivo

# Etiquetas de las emociones
emotion_labels = ['Enojo', 'Asco', 'Miedo', 'Felicidad', 'Tristeza', 'Sorpresa', 'Neutral']

# Iniciar la captura de video
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = detector(gray)
    
    for face in faces:
        # Extraer los 68 puntos faciales
        landmarks = predictor(gray, face)
        points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
        # Extraer la región de la cara
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # Verificar que las dimensiones de la cara sean válidas
        if x1 >= 0 and y1 >= 0 and x2 <= gray.shape[1] and y2 <= gray.shape[0]:
            face_img = gray[y1:y2, x1:x2]

            # Verificar si la cara extraída no está vacía
            if face_img.size > 0:
                face_img = cv2.resize(face_img, (48, 48))
                face_img = face_img.reshape(1, 48, 48, 1)

                # Predecir la emoción
                prediction = model.predict(face_img)
                emotion = emotion_labels[np.argmax(prediction)]

                # Mostrar la emoción predicha en la imagen
                cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen con puntos faciales y emociones
    cv2.imshow('Emotion Detector', frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
