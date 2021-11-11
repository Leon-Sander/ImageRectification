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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import urllib

app = Flask(__name__)

model = crease()
model.load_state_dict(torch.load('models/pretrained/crease_monster_best.pkl'))
model.eval()

'''@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')'''

@app.route("/", methods=['POST'])
def rectify():
    image_file = request.files['imagefile']
    image_path = "./server_folder/images/" + image_file.filename
    image_file.save(image_path)
    image_file = cv2.imread(image_path)
    preprocessed_img = preprocess.preprocess_img(image_file)
    bm_prediction = model.forward(preprocessed_img)
    unwarped_img = model.unwarp_image(preprocessed_img,bm_prediction.transpose(1,2).transpose(2,3))
    #pngImageB64String = "data:image/png;base64,"
    #pngImageB64String += base64.b64encode(output.getvalue()).decode('utf8')
    
    img = io.BytesIO()
    plt.imshow(unwarped_img)
    plt.savefig(img, format='png')  # save figure to the buffer
    img.seek(0)  # rewind your buffer
    #plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode()) # base64 encode & URL-escape
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('index.html', plot_url=plot_url)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug = True)