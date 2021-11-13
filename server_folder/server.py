import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import io
from flask import Flask, request, render_template, send_file
from models.full_model import crease
from server_folder import preprocess
import torch
from PIL import Image
import cv2
import logging
import base64

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
model = crease()
model.load_state_dict(torch.load('models/pretrained/crease_monster_best.pkl'))
model.eval()

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/plot", methods=['POST'])
def rectify():
    image_file = request.files['imagefile']
    image_path = "./server_folder/images/" + image_file.filename
    image_file.save(image_path)
    image_file = cv2.imread(image_path)
    preprocessed_img = preprocess.preprocess_img(image_file)
    bm_prediction = model.forward(preprocessed_img)
    unwarped_img = model.unwarp_image(preprocessed_img,bm_prediction.transpose(1,2).transpose(2,3))
    unwarped_img = unwarped_img * 255
    #cv2.resize(unwarped_img, )
    img = Image.fromarray(unwarped_img.astype('uint8'))
    
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    file_object.seek(0)  # rewind your buffer
    #app.logger.info(dir(img))

    data = base64.b64encode(file_object.getbuffer()).decode('ascii')

    return render_template('plot.html', plot_url=data)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug = False)