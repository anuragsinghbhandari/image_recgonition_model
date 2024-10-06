import cv2
import numpy as np
from retinaface import RetinaFace
from deepface import DeepFace

# Load the group image and detect faces using RetinaFace
image_path = 'test.jpg'
image = cv2.imread(image_path)

# Use RetinaFace to detect faces
faces = RetinaFace.detect_faces(image)

# Initialize known face embeddings (from the database)
known_faces = []
known_profiles = []


# Process each detected face in the group image
for i, (key, face_data) in enumerate(faces.items()):
    # Get face bounding box
    bbox = face_data['facial_area']
    face_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # Crop the detected face
    
    # Resize the face crop to a suitable size (DeepFace handles resizing internally)
    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    
    # Get face embedding using DeepFace (ArcFace model)
    face_in_group = DeepFace.represent(face_crop_rgb, model_name="ArcFace", enforce_detection=False)
    
    if len(face_in_group) > 0:
        unknown_embedding = face_in_group[0]['embedding']  # Extract embedding
        
        # Compare detected face with known faces
        for j, known_embedding in enumerate(known_faces):
            # Calculate cosine similarity
            similarity = np.dot(known_embedding, unknown_embedding) / (np.linalg.norm(known_embedding) * np.linalg.norm(unknown_embedding))
            
            # Set a threshold for matching
            if similarity > 0.4:
                print(f"Face {i} matches with profile {known_profiles[j]} with similarity {similarity:.2f}")
            # else:
            #     print(f"Face {i} does not match with profile {known_profiles[j]}")
    else:
        print(f"Face {i} could not be processed for embeddings.")
