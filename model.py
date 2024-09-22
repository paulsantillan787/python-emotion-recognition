import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Cargar el dataset FER2013
data = pd.read_csv('fer2013.csv')

# Preprocesamiento de datos
def preprocess_data(data):
    X = []
    y = []
    for index, row in data.iterrows():
        pixels = np.array(row['pixels'].split(' '), 'float32')
        pixels = pixels.reshape(48, 48, 1)
        X.append(pixels)
        y.append(row['emotion'])
    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=7)
    return X, y

X, y = preprocess_data(data)

# Dividir en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo CNN
model = Sequential()

# Capas convolucionales
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capas densas
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=64)

# Guardar el modelo
model.save('emotion_model.h5')
