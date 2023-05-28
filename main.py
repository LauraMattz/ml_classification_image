import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image


# Configurações
data_dir = './data'
train_dir = os.path.join(data_dir, 'train')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

img_height, img_width = 150, 150
batch_size = 32
num_classes = 2  # Altere com o número de classes que houver
epochs = 25

# Gerador de dados
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # 80% treinamento e 20% validação
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Definir modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treinamento
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator)

# Salvar modelo
model.save('meu_modelo.h5')

print("Modelo treinado e salvo com sucesso!")


def predict_image(model, image_path, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255
    prediction = model.predict(img_array)
    return prediction


def get_class_label(prediction, class_indices):
    class_label = None
    max_prob = np.max(prediction)
    for label, index in class_indices.items():
        if prediction[0][index] == max_prob:
            class_label = label
            break
    return class_label


# Carregar modelo treinado
model = load_model('meu_modelo.h5')

# Criar gerador de dados para conjunto de treinamento
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    './data',
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical')

# Obter índices de classes do conjunto de treinamento
class_indices = train_generator.class_indices

# Caminho para a imagem a ser prevista
image_path = './data/1.jfif'
# Prever a classe da imagem
prediction = predict_image(model, image_path,  150,150)
class_label = get_class_label(prediction, class_indices)

# Gráfico de Acurácia
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(range(1, epochs+1), train_acc, label='Train Accuracy')
plt.plot(range(1, epochs+1), val_acc, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Gráfico de Perda
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(range(1, epochs+1), train_loss, label='Train Loss')
plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Imprimir valores do último epoch
last_epoch = epochs
train_acc_last = train_acc[last_epoch - 1]
val_acc_last = val_acc[last_epoch - 1]
train_loss_last = train_loss[last_epoch - 1]
val_loss_last = val_loss[last_epoch - 1]

print(f"Último Epoch - Train Accuracy: {train_acc_last:.4f}")
print(f"Último Epoch - Validation Accuracy: {val_acc_last:.4f}")
print(f"Último Epoch - Train Loss: {train_loss_last:.4f}")
print(f"Último Epoch - Validation Loss: {val_loss_last:.4f}")

# Interpretação dos resultados
if val_acc_last > 0.5:
    print("O modelo foi capaz de obter uma acurácia razoável na validação.")
else:
    print("O modelo teve dificuldade em generalizar os dados e obteve baixa acurácia na validação.")

if train_loss_last < val_loss_last:
    print("Existe uma possível indicação de overfitting, pois a perda no treinamento é menor do que a perda na validação.")
else:
    print("Os resultados de perda do treinamento e da validação estão equilibrados.")


def predict_image(model, image_path, img_height, img_width):
    img = Image.open(image_path)
    img = img.resize((img_height, img_width))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction


def get_class_label(prediction, class_indices):
    class_label = None
    max_prob = np.max(prediction)
    for label, index in class_indices.items():
        if prediction[0][index] == max_prob:
            class_label = label
            break
    return class_label


# Carregar modelo treinado
model = load_model('meu_modelo.h5')

# Caminhos para as imagens a serem previstas
image_paths = [
    './data/1.jfif',
    './data/2.jfif',
    './data/3.jfif',
    './data/4.jfif',
    './data/5.jfif',
    './data/6.jpg'
    # Adicione mais caminhos de imagens conforme necessário
]

# Obter índices de classes do conjunto de treinamento
class_indices = {'cachorro': 0, 'cavalo': 1 }  # Substitua com seus índices reais

# Configurar a grade de subplots
num_images = len(image_paths)
num_cols = 2
num_rows = (num_images + num_cols - 1) // num_cols

# Prever a classe das imagens e exibir
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))

for i, image_path in enumerate(image_paths):
    prediction = predict_image(model, image_path, 150, 150)
    class_label = get_class_label(prediction, class_indices)
    prob = np.max(prediction) * 100
    
    # Exibir a imagem e o resultado
    img = Image.open(image_path)
    ax = axes[i // num_cols, i % num_cols]
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Classe: {class_label} | Probabilidade: {prob:.2f}%")

# Ajustar o espaçamento entre os subplots
plt.tight_layout()

# Exibir a figura
plt.show()
