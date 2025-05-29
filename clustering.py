import os
import shutil
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

def extract_face_embedding(image_path):
    try:
        resp = DeepFace.represent(img_path=image_path, model_name="Facenet", detector_backend='retinaface')
        if resp:
            return resp[0]['embedding']
    except Exception as e:
        print(f"Embedding extraction failed for {image_path}: {e}")
    return None

def cluster_faces(embeddings):
    normalized_embeddings = normalize(np.array(embeddings))
    model = DBSCAN(metric='cosine', eps=0.5, min_samples=1)
    labels = model.fit_predict(normalized_embeddings)
    return labels

def create_albums(image_paths, labels, output_folder):
    for idx, label in enumerate(labels):
        folder = os.path.join(output_folder, f"Person_{label + 1}")
        os.makedirs(folder, exist_ok=True)
        shutil.copy(image_paths[idx], os.path.join(folder, os.path.basename(image_paths[idx])))

# ✅ This is the wrapper function used in Flask:
def process_images(image_paths, output_folder):
    embeddings = []
    valid_paths = []
    
    for path in image_paths:
        if not path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        embedding = extract_face_embedding(path)
        if embedding is not None:
            embeddings.append(embedding)
            valid_paths.append(path)
    
    if embeddings:
        labels = cluster_faces(embeddings)
        create_albums(valid_paths, labels, output_folder)
    else:
        raise Exception("No valid face embeddings found.")
