from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import numpy as np
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os

base = 'Skin cancer'

model_path = os.path.join(base, 'mode_skincancer.h5')
model = load_model(model_path)


def show_output(image):
    img = cv2.resize(cv2.imread(image,cv2.IMREAD_GRAYSCALE),(75,75))
    img =  np.asarray(img).reshape(-1,75,75,1)
    out = np.round(model.predict(img))
    lesion_type_dict = {
    'Melanocytic nevi':np.array([[1, 0, 0, 0,0, 0, 0]]),
    'Melanoma':np.array([[0, 1, 0, 0,0, 0, 0]]),
    'Benign keratosis-like lesions ':np.array([[0, 0, 1, 0,0, 0, 0]]),
    'Basal cell carcinoma':np.array([[0, 0, 0, 1,0, 0, 0]]),
    'Actinic keratoses':np.array([[0, 0, 0, 0,1, 0, 0]]),
    'Vascular lesions':np.array([[0, 0, 0, 0,0, 1, 0]]),
    'Dermatofibroma':np.array([[0. ,0. ,0., 0. ,0. ,0. ,1.]])}

    for key,value in lesion_type_dict.items():
        if np.array_equal(out,value):
            return key
    return "No Match found"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
 app = Flask(__name__, template_folder = os.path.join(base,"templates"),static_folder= os.path.join(base,'static'))
UPLOAD_FOLDER = os.path.join(base,'upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

run_with_ngrok(app)   #starts ngrok when the app is run
@app.route("/",methods=['GET', 'POST'])
def home():
    out = ''
    if request.method == 'POST':
        image = request.files['main_img']
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',filename=filename))
    
    return render_template('base.html',out=out)

@app.route('/show/<filename>')
def uploaded_file(filename):
    image = os.path.join(UPLOAD_FOLDER,filename)
    out = show_output(image)
    return render_template('base.html', out=out,filename=filename)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


app.run()
