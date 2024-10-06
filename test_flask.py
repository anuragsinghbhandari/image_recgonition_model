from flask import Flask, request, jsonify
import cv2
from retinaface import RetinaFace
from deepface import DeepFace
import numpy as np

app = Flask(__name__)

# Endpoint to handle face detection and embedding
@app.route('/process-image', methods=['POST'])
def process_image():
    # Check if an image file is included in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    
    # Convert the uploaded image to a numpy array
    image_data = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(image)
    
    if not faces:
        return jsonify({"error": "No faces detected"}), 400
    
    embeddings = []
    
    for i, (key, face_data) in enumerate(faces.items()):
        # Get face bounding box
        bbox = face_data['facial_area']
        face_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # Crop the detected face
        
        # Convert BGR to RGB (DeepFace expects RGB)
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Get face embedding using DeepFace
        profile_embed = DeepFace.represent(face_crop_rgb, model_name="ArcFace", enforce_detection=False)
        embeddings.append(profile_embed[0]['embedding'])
        break  # Process only the first face detected
    
    # Return the embedding in JSON format
    return jsonify({"embedding": embeddings[0]})

if __name__ == '__main__':
    # Ensure debug mode is off in production
    app.run(debug=True)
