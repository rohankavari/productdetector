from flask import Flask, render_template, request, redirect, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import random
from helper import detect_objects_yolo,get_file_paths
app = Flask(__name__)

UPLOAD_FOLDER = 'static\\uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{random.randint(1,1000)}_img.{file.filename[-3:]}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = cv2.imread(filepath)
            detected_image = detect_objects_yolo(image,filename)  
            cv2.imwrite(filepath, detected_image)
            
            return render_template('detect.html', filename=filename)
    
    return render_template('detect.html')

@app.route('/detect',methods=['POST'])
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{random.randint(1,1000)}_img.{file.filename[-3:]}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            image = cv2.imread(filepath)
            detected_image = detect_objects_yolo(image,filename)  
            cv2.imwrite(filepath, detected_image)
        data=get_file_paths(f"static/result/{filename}",request.host_url)
        return jsonify(data)
if __name__ == '__main__':
    app.run(port=8000)
