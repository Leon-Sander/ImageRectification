import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import io
from flask import Flask, request, render_template, Response, send_file
from models.full_model import crease
from server_folder import preprocess
import torch
from PIL import Image
import cv2
import logging
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
model = crease()
model.load_state_dict(torch.load('models/pretrained/crease_monster_best.pkl'))
model.eval()

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def rectify():
    image_file = request.files['imagefile']
    image_path = "./server_folder/images/" + image_file.filename
    image_file.save(image_path)
    image_file = cv2.imread(image_path)
    preprocessed_img = preprocess.preprocess_img(image_file)
    bm_prediction = model.forward(preprocessed_img)
    unwarped_img = model.unwarp_image(preprocessed_img,bm_prediction.transpose(1,2).transpose(2,3))
    #app.logger.info(unwarped_img)
    unwarped_img = unwarped_img * 255
    img = Image.fromarray(unwarped_img.astype('uint8'))
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    file_object.seek(0)  # rewind your buffer

    return send_file(file_object, mimetype='image/PNG')
    #return render_template('index.html', plot_url=send_file(file_object, mimetype='image/PNG'))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug = True)