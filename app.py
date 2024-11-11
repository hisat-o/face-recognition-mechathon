from flask import Flask, request, render_template, redirect, url_for, flash, send_file
import numpy as np
import os
import cv2
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import cosine
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = 'your_secret_key'  # Replace with a strong key for production

# Load model and embeddings
data = np.load('face_embeddings.npz')
embedded_X = data['arr_0']
labels = data['arr_1']

detector = MTCNN()
embedder = FaceNet()

# Calculate average embedding for the target person
def get_average_embedding(person_name, embedded_X, labels):
    person_embeddings = embedded_X[labels == person_name]
    return np.mean(person_embeddings, axis=0)

# Function to process and find best match from files
def find_best_match_from_files(uploaded_files, target_person_name):
    target_embedding = get_average_embedding(target_person_name, embedded_X, labels)
    best_score = -1
    best_image_path = None

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        img = cv2.imread(file_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)
        if len(results) == 0:
            continue

        x, y, w, h = results[0]['box']
        face = img_rgb[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (160, 160))

        test_embedding = embedder.embeddings(np.expand_dims(face_resized, axis=0))[0]
        similarity = 1 - cosine(test_embedding, target_embedding)

        if similarity > best_score:
            best_score = similarity
            best_image_path = file_path

    return best_image_path, best_score

# Function to process and find best match from a single input image
def find_best_match_from_image(input_image, uploaded_files):
    best_score = -1
    best_image_path = None

    # Convert the uploaded image to embedding
    input_img = cv2.imread(input_image)
    if input_img is None:
        return None, None
    
    input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(input_img_rgb)
    if len(results) == 0:
        return None, None

    x, y, w, h = results[0]['box']
    input_face = input_img_rgb[y:y + h, x:x + w]
    input_face_resized = cv2.resize(input_face, (160, 160))

    input_embedding = embedder.embeddings(np.expand_dims(input_face_resized, axis=0))[0]

    # Loop through uploaded files and find the best match
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        img = cv2.imread(file_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)
        if len(results) == 0:
            continue

        x, y, w, h = results[0]['box']
        face = img_rgb[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (160, 160))

        test_embedding = embedder.embeddings(np.expand_dims(face_resized, axis=0))[0]
        similarity = 1 - cosine(input_embedding, test_embedding)

        if similarity > best_score:
            best_score = similarity
            best_image_path = file_path

    return best_image_path, best_score

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        target_person_name = request.form['person_name']
        uploaded_files = request.files.getlist('files')
        input_image = request.files.get('input_image')

        if not uploaded_files and not input_image:
            flash("No files or image uploaded.")
            return redirect(url_for('index'))

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        if input_image:
            # If the user uploaded a single image, find the best match from the folder
            input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(input_image.filename))
            input_image.save(input_image_path)
            best_image_path, best_score = find_best_match_from_image(input_image_path, uploaded_files)
        else:
            # If the user wants to find best match by name
            best_image_path, best_score = find_best_match_from_files(uploaded_files, target_person_name)

        for file in uploaded_files:
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            if temp_path != best_image_path:
                os.remove(temp_path)

        if best_image_path:
            flash(f"Best match found with confidence score: {best_score:.2f}")
            # Use os.path.relpath to handle paths more reliably
            best_image_relative_path = os.path.relpath(best_image_path, app.config['UPLOAD_FOLDER'])
            return render_template('index.html', best_image=best_image_relative_path)

        flash("No match found or face detected.")
        return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
