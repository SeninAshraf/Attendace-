from flask import Flask, render_template, request, jsonify
import sqlite3
import cv2
import os
import numpy as np
from datetime import date

app = Flask(__name__)

# Create database if not exists
DB_PATH = "face-reco.db"
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, date TEXT)")
    con.commit()
    con.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')  # Ensure your HTML file is named index.html and in a "templates" folder

@app.route('/login', methods=['POST'])
def login():
    password = request.form.get("password")
    if password == "123":
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid Password"})

@app.route('/train', methods=['POST'])
def train():
    name = request.form.get("name")
    capture_count = int(request.form.get("count", 100))

    datasets = 'datasets'
    path = os.path.join(datasets, name)
    if not os.path.isdir(path):
        os.makedirs(path)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    webcam = cv2.VideoCapture(0)
    count = 1

    while count <= capture_count:
        ret, frame = webcam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resize = cv2.resize(face, (130, 100))
            cv2.imwrite(f'{path}/{count}.png', face_resize)
            count += 1
        cv2.imshow('Training', frame)
        if cv2.waitKey(10) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()
    return jsonify({"status": "success", "message": "Training Completed"})

@app.route('/record_attendance', methods=['POST'])
def record_attendance():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    datasets = 'datasets'
    (images, labels, names, id) = ([], [], {}, 0)

    for subdir in os.listdir(datasets):
        names[id] = subdir
        subject_path = os.path.join(datasets, subdir)
        for filename in os.listdir(subject_path):
            path = os.path.join(subject_path, filename)
            images.append(cv2.imread(path, 0))
            labels.append(id)
        id += 1

    images, labels = np.array(images), np.array(labels)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)

    webcam = cv2.VideoCapture(0)
    detected = False

    while True:
        ret, frame = webcam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resize = cv2.resize(face, (130, 100))
            prediction = model.predict(face_resize)

            if prediction[1] < 800:
                name = names[prediction[0]]
                con = sqlite3.connect(DB_PATH)
                cursor = con.cursor()
                cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, str(date.today())))
                if not cursor.fetchall():
                    cursor.execute("INSERT INTO attendance (name, date) VALUES (?, ?)", (name, str(date.today())))
                    con.commit()
                con.close()
                detected = True
                break

        if detected:
            break

    webcam.release()
    cv2.destroyAllWindows()
    return jsonify({"status": "success", "message": "Attendance Recorded"})

@app.route('/reports', methods=['GET'])
def reports():
    con = sqlite3.connect(DB_PATH)
    cursor = con.execute("SELECT * FROM attendance")
    data = [{"id": row[0], "name": row[1], "date": row[2]} for row in cursor.fetchall()]
    con.close()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)