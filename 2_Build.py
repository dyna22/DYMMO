import os
import cv2
import numpy as np
from keras.models import load_model


def f(pic, required_size=(160, 160)):
    image = cv2.imread(pic)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, required_size)
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
    with open('DS\\dataset.txt', 'a') as input:
        ds = open('DS\\dataset.txt', 'r').readlines()
        for subdir in os.listdir(directory):
            count = 0
            for line in ds:  # line\n
                if line == str(subdir + '\n'):
                    print(subdir + ' is exist')
                    count += 1
                    continue
            if count is not 0:
                continue
            input.write(str(subdir) + "\n")
            path = directory + subdir + '/'
            if not os.path.isdir(path):
                continue
            faces = load_faces(path)
            labels = [subdir for _ in range(len(faces))]
            print('>>loaded %d images for class: %s' % (len(faces), subdir))
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


train_X, train_y = load_dataset('DS/people/')

model = load_model('DS/face_net_keras.h5')

newTrainX = list()
for face_pixels in train_X:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
'''
if os.path.isdir('DS/faces-embeddings.npz'):
    print('does not exist model')
    np.savez_compressed('DS/faces-embeddings.npz', newTrainX, train_y)

elif not os.path.isdir('DS/faces-embeddings.npz'):
    print('exist model')
'''
data = np.load('DS/faces-embeddings.npz')

train_X = np.concatenate([data['arr_0'], newTrainX])
train_y = np.concatenate([data['arr_1'], train_y])

np.savez('DS/faces-embeddings.npz', train_X, train_y)
