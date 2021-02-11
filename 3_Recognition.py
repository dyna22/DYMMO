from sklearn.preprocessing import Normalizer
import cv2
from keras.models import load_model
import numpy as np


def get_embedding(model, image):
    face_pixels = image.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    pre = model.predict(samples)
    return pre[0]


def predict(faces_embeddings, labels, new_face_emb):
    in_encoder = Normalizer(norm='l2')
    faces_embeddings = in_encoder.transform(faces_embeddings)
    new_face_emb = in_encoder.transform(new_face_emb)

    face_distance = np.linalg.norm(faces_embeddings - new_face_emb, axis=1)

    name = 'UNKNOWN'  # with this we don't need unknown file with strange people
    matches = list(face_distance <= 0.7)  # threshold
    if True in matches:
        matched_index = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matched_index:
            name = labels[i]
            counts[name] = counts.get(name, 0) + 1

        name = max(counts, key=counts.get)
    return name


def predict2(faces_embeddings, labels, new_face_emb):
    # normalize input vectors 34an diff ranges
    in_encoder = Normalizer(norm='l2')
    faces_embeddings = in_encoder.transform(faces_embeddings)
    new_face_emb = in_encoder.transform(new_face_emb)
    # euclidean distance
    face_distance = np.linalg.norm(faces_embeddings - new_face_emb, axis=1)
    index = np.argmin(face_distance)

    return labels[index]


model = load_model('DS/face_net_keras.h5')
data = np.load('DS/faces-embeddings.npz')
faces_embeddings, labels = data['arr_0'], data['arr_1']

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cascade = cv2.CascadeClassifier("cascade.xml")
cv2.namedWindow("RECO")
while True:
    frame, image = cam.read()
    faces = cascade.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in faces:
        croped = image[y:y + h, x:x + w]
        gray = cv2.cvtColor(croped, cv2.COLOR_BGR2GRAY)
        # contrast limited adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        colored = cv2.cvtColor(cl, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(colored, (160, 160))
        img_array = np.array(resized)
        new_face_emb = get_embedding(model, img_array)
        new_face_emb = np.asarray(new_face_emb)
        new_face_emb = new_face_emb.reshape(1, -1)
        predict_name = predict2(faces_embeddings, labels, new_face_emb)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image, predict_name, (x + 10, y + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
    cv2.imshow('RECO', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

