from deepface import DeepFace
import os
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import pickle
import json
import cv2
import glob 

class face_recognition:
    def create_db(self):
        dataset_path = "dataset/"
        people = os.listdir(dataset_path)

        images = []
        labels = []

        for person in people:
            person_folder = os.path.join(dataset_path, person)
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                images.append(image_path)
                labels.append(person)

        embeddings = []
        for image in images:
            embedding = DeepFace.represent(img_path=image, model_name="Facenet")
            embeddings.append(embedding)
        labeled_embeddings = list(zip(embeddings, labels))
        with open('database/embeddings.pkl', 'wb') as file:
            pickle.dump(labeled_embeddings, file)
        with open('check.json', 'w') as fw:
            json.dump(labeled_embeddings, fw, indent= 4)
    #create_db() # update db

    def create_new_face(self, person_name, image):
        folder = f'dataset/{person_name}'
        if not os.path.exists(folder):
            os.makedirs(folder)
            count = 1
        else:
            files = glob.glob(folder + '/*')
            count = len(files) + 1
        image_path = os.path.join(folder, f'{count}.jpg')
        cv2.imwrite(image_path, image)
        self.create_db()

    def find_person(self, image_path):
        with open('database/embeddings.pkl', 'rb') as file:
            labeled_embeddings = pickle.load(file)
        new_embedding = DeepFace.represent(img_path=image_path, model_name="Facenet")[0]["embedding"]
        min_dist = float('inf')
        best_label = None
        for image in labeled_embeddings:
            image_info = image[0][0]
            label = image[1] 
            embedding = image_info['embedding']
            dist = cosine(new_embedding, embedding)
            if dist < min_dist:
                min_dist = dist
                best_label = label
        return best_label, min_dist


