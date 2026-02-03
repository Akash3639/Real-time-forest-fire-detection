from flask import Blueprint, render_template, jsonify, request, Response
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from datetime import datetime, timedelta
import os
from threading import Thread
import torch
import time
from pygame import mixer
import base64
from itertools import zip_longest
import matplotlib.pyplot as plt

from database import db, Fire_Alerts, Fire_Location
from App import create_app

# =========================================================
# YOLOv5 LEGACY MODEL LOADING (FOR yolocff.pt)
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model1 = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="Models/yolocff.pt",
    trust_repo=True
)
model1.to(device)
classes = model1.names

# =========================================================
# GLOBAL VARIABLES
# =========================================================
neg = 0
switch = 0
rec = 0
rec_frame = None
camera = None
fire_detected = False

os.makedirs("static/shots", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/video", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)

# =========================================================
# VIDEO RECORDING
# =========================================================
def record(out):
    global rec_frame
    while rec:
        time.sleep(0.05)
        out.write(rec_frame)

# =========================================================
# YOLO INFERENCE
# =========================================================
def score_frame(frame):
    results = model1(frame)
    labels = results.xyxy[0][:, -1].cpu().numpy()
    cords = results.xyxy[0][:, :-1].cpu().numpy()
    return labels, cords

def class_to_label(x):
    return classes[int(x)]

def plot_boxes(results, frame):
    global fire_detected
    labels, cords = results

    for i in range(len(labels)):
        x1, y1, x2, y2, conf = cords[i][:5]
        if conf < 0.25:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            class_to_label(labels[i]),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        if not fire_detected:
            mixer.init()
            mixer.Sound("fire_alarm.ogg").play()

            now = datetime.now()
            img_path = f"static/shots/fire_{now.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(img_path, frame)

            with open(img_path, "rb") as f:
                encoded_img = base64.b64encode(f.read()).decode()

            alert = Fire_Alerts(
                date=str(now.date()),
                time=str(now.time()),
                image_path=encoded_img
            )
            db.session.add(alert)
            db.session.commit()

            fire_detected = True
        break

    return frame

# =========================================================
# CAMERA STREAM
# =========================================================
def gen_frames(app):
    global camera, rec_frame
    with app.app_context():
        while True:
            if camera is None:
                continue

            success, frame = camera.read()
            if not success:
                continue

            if switch:
                frame = plot_boxes(score_frame(frame), frame)

            if rec:
                rec_frame = frame

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# =========================================================
# IMAGE CLASSIFICATION (CNN)
# =========================================================
def predict_label(img_path):
    model = load_model("Models/fire_smoke_and_nonfire_detection.h5", compile=False)
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    return ["Smoke", "Fire", "No Fire"][np.argmax(pred)]

# =========================================================
# FLASK BLUEPRINT
# =========================================================
View = Blueprint("View", __name__)

@View.route("/")
def Home():
    return render_template("index.html")

@View.route("/About")
def About():
    return render_template("About.html")

@View.route("/Prediction")
def Prediction():
    return render_template("Prediction.html")

@View.route("/submit", methods=["POST"])
def get_output():
    img = request.files["my_image"]
    img_path = f"static/uploads/{img.filename}"
    img.save(img_path)
    result = predict_label(img_path)
    return render_template(
        "Prediction.html",
        prediction=result,
        img_path=img_path
    )

@View.route("/FireAlerts")
def FireAlerts():
    app = create_app()
    with app.app_context():
        alerts = Fire_Alerts.query.all()
        locations = Fire_Location.query.all()
        data = list(zip_longest(alerts, locations))
    return render_template("FireAlerts.html", combined_data=data)

@View.route("/ModelTesting")
def ModelTesting():
    history = np.load("my_history.npy", allow_pickle=True).item()

    # Accuracy plot
    plt.figure()
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig("static/graphs/Acc_plot.png")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.savefig("static/graphs/Loss_plot.png")
    plt.close()

    return render_template(
        "ModelTesting.html",
        acc_plot_url="static/graphs/Acc_plot.png",
        loss_plot_url="static/graphs/Loss_plot.png"
    )

@View.route("/Location")
def Location():
    return render_template("Location.html")

@View.route("/LiveMonitor")
def LiveMonitor():
    return render_template("LiveMonitor.html")

@View.route("/video_feed")
def video_feed():
    app = create_app()
    return Response(
        gen_frames(app),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@View.route("/requests", methods=["POST", "GET"])
def tasks():
    global switch, camera, rec, out

    if request.method == "POST":
        if request.form.get("stop"):
            if switch == 0:
                camera = cv2.VideoCapture(0)
                switch = 1
            else:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

        elif request.form.get("rec"):
            rec = not rec
            if rec:
                now = datetime.now()
                out = cv2.VideoWriter(
                    f"static/video/vid_{now.strftime('%Y%m%d_%H%M%S')}.avi",
                    cv2.VideoWriter_fourcc(*"XVID"),
                    20.0,
                    (640, 480)
                )
                Thread(target=record, args=(out,)).start()
            else:
                out.release()

    return render_template("LiveMonitor.html")
