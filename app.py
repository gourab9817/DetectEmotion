import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import io
from mtcnn import MTCNN
import os
from datetime import datetime

# 1) Import environment-loading library
from dotenv import load_dotenv

# 2) Import Supabase client
from supabase import create_client, Client

# 3) Load environment variables from .env
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'shamstabrez'  # Update to your own secret

# 4) Retrieve Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 5) Create a Supabase client using the environment variables
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 6) Load your trained model
model_path = os.path.join("model", "RAFDB_Custom.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = tf.keras.models.load_model(model_path)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = (48, 48)
detector = MTCNN()

# We'll keep track of the current session; logs are not stored locally.
current_session_id = None

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def generate_session_id():
    """Generate a unique session ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d%H%M%S")

def detect_and_classify(frame):
    """
    Detect faces using MTCNN, then classify each face with the loaded model.
    Returns the frame with bounding boxes & emotion text, plus top-3 emotions.
    """
    faces = detector.detect_faces(frame)
    detected_faces = []
    if faces:
        for face in faces:
            x, y, w, h = face['box']
            # Ensure bounding box is within valid bounds
            x, y = max(0, x), max(0, y)
            cropped_face = frame[y:y+h, x:x+w]

            if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                # Preprocess face for the model
                face_rgb = cv2.resize(cropped_face, IMG_SIZE)
                face_array = tf.keras.preprocessing.image.img_to_array(face_rgb) / 255.0
                face_array = np.expand_dims(face_array, axis=0)

                # Predict emotions
                predictions = model.predict(face_array)[0]
                top_indices = np.argsort(predictions)[-3:][::-1]  # Top 3
                top_emotions = [
                    (class_labels[i], float(round(predictions[i] * 100, 2)))
                    for i in top_indices
                ]

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                for i, (emotion, percentage) in enumerate(top_emotions):
                    text = f"{emotion} ({percentage}%)"
                    cv2.putText(
                        frame,
                        text,
                        (x, y - (i * 20) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

                detected_faces.extend(top_emotions)

    return frame, detected_faces

# ---------------------------
# SUPABASE HELPERS
# ---------------------------
def save_emotion_data_to_supabase(session_id, emotions):
    """
    Insert each detected emotion into the 'emotion_logs' table on Supabase.
    'emotions' is a list of tuples like: [(emotionName, percentage), ...].
    """
    timestamp = datetime.now().isoformat()
    rows_to_insert = []

    for (emotion, percentage) in emotions:
        rows_to_insert.append({
            "session_id": session_id,
            "timestamp": timestamp,
            "emotion": emotion,
            "percentage": percentage,
        })

    # Insert in bulk if there's anything
    if rows_to_insert:
        supabase.table("emotion_logs").insert(rows_to_insert).execute()

def fetch_emotion_data_from_supabase(session_id):
    """
    Query all records for the given session_id from the 'emotion_logs' table.
    Returns a list of dicts: [{session_id, timestamp, emotion, percentage}, ...].
    """
    response = supabase.table("emotion_logs") \
                       .select("*") \
                       .eq("session_id", session_id) \
                       .execute()
    return response.data if response.data else []

def calculate_average_emotions(session_id):
    """
    Compute average percentage for each emotion in a single session,
    sorted descending by average percentage.
    Returns a list of tuples: [(emotion, avgPct), ...].
    """
    records = fetch_emotion_data_from_supabase(session_id)
    if not records:
        return []

    emotion_sums = {}
    emotion_counts = {}

    for row in records:
        emotion = row["emotion"]
        pct = float(row["percentage"])
        if emotion not in emotion_sums:
            emotion_sums[emotion] = 0.0
            emotion_counts[emotion] = 0
        emotion_sums[emotion] += pct
        emotion_counts[emotion] += 1

    average_emotions = []
    for emotion, total in emotion_sums.items():
        avg = round(total / emotion_counts[emotion], 2)
        average_emotions.append((emotion, avg))

    average_emotions.sort(key=lambda x: x[1], reverse=True)
    return average_emotions

def count_records_for_session(session_id):
    """Return the total count of detection logs for the given session_id."""
    records = fetch_emotion_data_from_supabase(session_id)
    return len(records)

# ---------------------------
# FLASK ROUTES
# ---------------------------
@app.route('/')
def index():
    """The main page rendering function."""
    return render_template(
        'index.html',
        top_emotions=None,
        img_base64=None,
        show_upload=True,
        show_camera=False,
        initial_image=True
    )

@app.route('/classify', methods=['POST'])
def classify_image():
    """
    Optional route for single-image classification (non-video).
    You can remove this if not needed.
    """
    image = request.files['image']
    img = Image.open(image)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    processed_frame, top_emotions = detect_and_classify(img)
    _, buffer = cv2.imencode('.png', processed_frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return render_template(
        'index.html',
        top_emotions=top_emotions,
        img_base64=img_base64,
        show_upload=True,
        show_camera=False,
        initial_image=False
    )

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """
    Receives base64 frames from the client, classifies emotions, saves in Supabase,
    and returns the processed frame + emotions in JSON format.
    """
    global current_session_id

    data = request.json
    image_data = data.get("image")
    session_id = data.get("session_id")

    if not image_data:
        return jsonify({"error": "No image data received"}), 400

    # If a session_id is provided, set that as current
    if session_id:
        current_session_id = session_id

    # Convert base64 image to a NumPy array
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Detect emotions
    processed_frame, detected_faces = detect_and_classify(frame)
    _, buffer = cv2.imencode(".jpg", processed_frame)
    processed_image_base64 = base64.b64encode(buffer).decode("utf-8")

    # Save results to Supabase
    if current_session_id and detected_faces:
        save_emotion_data_to_supabase(current_session_id, detected_faces)

    return jsonify({
        "processed_frame": processed_image_base64,
        "emotions": detected_faces
    })

@app.route('/session', methods=['POST'])
def handle_session():
    """
    Start or stop a session. 'start' => new session_id. 'stop' => compute average emotions.
    """
    global current_session_id

    data = request.json
    action = data.get("action")

    if action == "start":
        current_session_id = generate_session_id()
        return jsonify({"session_id": current_session_id})

    elif action == "stop" and data.get("session_id"):
        session_id = data.get("session_id")
        average_emotions = calculate_average_emotions(session_id)
        frames_analyzed = count_records_for_session(session_id)

        return jsonify({
            "session_id": session_id,
            "average_emotions": average_emotions,
            "frames_analyzed": frames_analyzed
        })

    return jsonify({"error": "Invalid action"}), 400

# ---------------------------
# LAUNCH
# ---------------------------
if __name__ == '__main__':
    # All local file-based logging is removed in favor of Supabase storage.
    app.run(host='0.0.0.0', port=7860)
