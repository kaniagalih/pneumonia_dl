from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import keras.utils as image
import numpy as np
import os

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder

dic = {0 : 'Normal', 1 : 'Pneumonia'}

model = load_model('model_vgg.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 224,224,3)
	p = model.predict(i)
	predict_index = np.argmax(p)
	confidence_score = np.max(p) * 100
	return dic[predict_index], confidence_score

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		file = request.files['my_image']
		file.save(os.path.join(app.config['UPLOAD'], file.filename))
		img = os.path.join(app.config['UPLOAD'], file.filename)
		p = predict_label(img)

	return render_template("index.html", prediction = p, confidence = p, img_path = img)


if __name__ =='__main__':
	app.run(debug = True)