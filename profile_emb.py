import cv2
from retinaface import RetinaFace
from deepface import DeepFace

# Load the group image and detect faces using RetinaFace
image_path = 'profile.jpg'
image = cv2.imread(image_path)

# Use RetinaFace to detect faces
faces = RetinaFace.detect_faces(image)

for i, (key, face_data) in enumerate(faces.items()):
    # Get face bounding box
    bbox = face_data['facial_area']
    face_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # Crop the detected face
    
    # Resize the face crop to a suitable size (DeepFace handles resizing internally)
    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    
    # Get face embedding using DeepFace (ArcFace model)
    profile_embed = DeepFace.represent(face_crop_rgb, model_name="ArcFace", enforce_detection=False)
    break
print(profile_embed[0]['embedding'])
