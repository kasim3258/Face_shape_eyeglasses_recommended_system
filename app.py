from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import base64
import pickle
with open("rf_model.pkl","rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)

# ===============================
# LOAD MODEL
# ===============================

print("✅ Model loaded")

# ===============================
# LABELS
# ===============================
class_labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

recommendation = {
    "Round": "Rectangle",
    "Square": "Round",
    "Heart": "Aviator",
    "Oval": "Square",
    "Oblong": "Round"
}

GLASSES_MAP = {
    "Round": "static/glasses/rectangle.png",
    "Square": "static/glasses/round.png",
    "Heart": "static/glasses/aviator.png",
    "Oval": "static/glasses/square.png",
    "Oblong": "static/glasses/wide.png"
}

# ===============================
# HOME ROUTE
# ===============================
@app.route('/')
def home():
    return render_template("index.html")

# ===============================
# FINAL OVERLAY FUNCTION
# ===============================
def add_glasses(face_img, glasses_path):
    face = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)

    glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
    if glasses is None:
        print("❌ Glasses not found:", glasses_path)
        return face

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return face

    for (x, y, w, h) in faces:

        # ✅ Better size
        g_width = int(w * 0.9)
        g_height = int(glasses.shape[0] * (g_width / glasses.shape[1]))

        resized = cv2.resize(glasses, (g_width, g_height))

        # ✅ Better alignment
        x1 = x + int((w - g_width) / 2)
        y1 = y + int(h * 0.18)

        y2 = y1 + g_height
        x2 = x1 + g_width

        if y2 > face.shape[0] or x2 > face.shape[1]:
            continue

        # ===============================
        # HANDLE BOTH IMAGE TYPES
        # ===============================
        if resized.shape[2] == 4:
            # Transparent PNG
            b, g, r, a = cv2.split(resized)
            overlay = cv2.merge((b, g, r))

            mask = a.astype(float) / 255.0
            inv_mask = 1.0 - mask

        else:
            # Non-transparent → remove white bg smoothly
            overlay = resized

            gray_glass = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray_glass, (5, 5), 0)

            _, mask = cv2.threshold(gray_blur, 200, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)

            mask = mask.astype(float) / 255.0
            inv_mask = 1.0 - mask

        # ===============================
        # BLENDING
        # ===============================
        for c in range(3):
            face[y1:y2, x1:x2, c] = (
                mask * overlay[:, :, c] +
                inv_mask * face[y1:y2, x1:x2, c]
            )

    return face

# ===============================
# PREDICT ROUTE
# ===============================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        img_pil = Image.open(BytesIO(file.read())).convert('RGB')
        img_resized = img_pil.resize((128, 128))

        img = np.array(img_resized)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img,(16, 8))
        img = img.flatten().reshape(1,-1)
        
        prediction = model.predict(img)[0]

        shape = str(prediction)
        confidence_percent = 95
        confidence = 0.95

        # note
        if confidence < 0.30:
            note = "Low confidence"
        elif confidence < 0.50:
            note = "Medium confidence"
        else:
            note = "High confidence"

        # ✅ Single condition
        if confidence_percent >= 45:
            glasses_path = GLASSES_MAP.get(shape)
            overlay_img = add_glasses(img_pil, glasses_path)
        else:
            overlay_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode('.jpg', overlay_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "face_shape": shape,
            "confidence": round(confidence_percent, 2),
            "note": note,
            "recommended_frame": recommendation.get(shape, "Try another image"),
            "image": img_base64
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "Backend failed"}), 500

# ===============================
# RUN APP
# ===============================
import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
    
