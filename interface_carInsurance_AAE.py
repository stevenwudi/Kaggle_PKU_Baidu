# CUDA_VISIBLE_DEVICES=4 FLASK_ENV=development FLASK_APP=interface_carInsurance_AAE.py flask run -p 5003
import os
import random
import base64
import io
import pprint

from PIL import Image
import numpy as np
import cv2
from flask import Flask
from flask import request, jsonify

from interface_utils import init_model, inference_detector, format_return_data, projective_distance_estimation_AAE, base64ToRGB

app = Flask(__name__)

model, cfg = init_model()


@app.route('/', methods=['POST'])
def hello():
    # - get `image_base64, fx, fy, cx, cy, ZRENDER, SCALE` from Flask service in terms of POST method.
    image_base64 = request.form.get('file')
    fx = float(request.form.get('fx'))
    fy = float(request.form.get('fy'))
    cx = float(request.form.get('cx'))
    cy = float(request.form.get('cy'))
    imgSizeX = int(request.form.get('imgSizeX'))
    imgSizeY = int(request.form.get('imgSizeY'))
    ZRENDER = float(request.form.get('ZRENDER'))
    SCALE = float(request.form.get('SCALE'))

    # Get the camera intrinsics here
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)

    image = base64ToRGB(image_base64)

    # Save a temporary image in the ./upload_imgs  folder
    img_i = random.randint(1, 100000)
    image_path = "/tmp/tmp_{}.jpg".format(img_i)
    cv2.imwrite(image_path, image)

    # Get the result from Kaggle competition model
    result = inference_detector(cfg, model, image_path)
    data = format_return_data(result)

    if data.shape[0] > 0:
        data = data[0]
        json = dict(
            code=0,
            msg='success',
            x1=data[0],
            y1=data[1],
            x2=data[2],
            y2=data[3],
            conf=data[4],
            rotation=list(data[5:8]),
            translation=list(data[8:]),
        )

        # Refine using AAE for the car 3D information here
        t_pred_x, t_pred_y, t_pred_z = projective_distance_estimation_AAE(json, image_path, camera_matrix, ZRENDER,
                                                                          SCALE)
        json['translation'] = list([t_pred_x, t_pred_y, t_pred_z])
        pprint.pprint(json)

    else:
        json = dict(
            status=1,
            msg='NO CAR'
        )
    # Finally we remove the image
    os.remove(image_path)
    return jsonify(json)
