import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from grad_cam import get_img_array, make_gradcam_heatmap, save_and_display_gradcam  


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

model = load_model('pneumonia_detection_model_VGGFace.h5')  # Load your trained model
last_conv_layer_name = 'conv2d_12'  # Specify the last convolutional layer

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            img_array = get_img_array(file_path, size=(224, 224))

            #img_array = tf.image.decode_image(tf.io.read_file(file_path))
            #img_array = tf.image.resize(img_array, (SIZE, SIZE))
            #img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            preds = model.predict(img_array)
            result = 'PNEUMONIA' if preds[0][0] > 0.5 else 'NORMAL'
            
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            save_and_display_gradcam(file_path, heatmap)

            return render_template('result.html', filename=filename, result=result)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

