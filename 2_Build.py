import os
import numpy as np
from PIL import Image
from keras.models import load_model


def f(pic, required_size=(160, 160)):
    image = Image.open(pic)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    image = Image.fromarray(pixels)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def load_faces(directory):
    faces = list()
    for filename in os.listdir(directory):
        path = directory + filename
        face = f(path)
        faces.append(face)
    return faces


def load_dataset(directory):
    X, y = list(), list()
    for subdir in os.listdir(directory):
        path = directory + subdir + '/'
        if not os.path.isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        # print('>>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    pre = model.predict(samples)
    return pre[0]


train_X, train_y = load_dataset('people/')

model = load_model('face_net_keras.h5')

newTrainX = list()
for face_pixels in train_X:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)

np.savez_compressed('faces-embeddings.npz', newTrainX, train_y)

data = np.load('faces-embeddings.npz')
train_X, train_y = data['arr_0'], data['arr_1']
# print('Dataset: people=%d' % (train_X.shape[0]))
