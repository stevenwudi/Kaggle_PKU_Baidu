#CUDA_VISIBLE_DEVICES=4 FLASK_ENV=development FLASK_APP=interface_AAE.py flask run -p 5003
import os
import random
import base64
import io

from PIL import Image
import numpy as np
import cv2
from flask import Flask
from flask import request, jsonify


from interface_utils import init_model, inference_detector, format_return_data, projective_distance_estimation_AAE

app = Flask(__name__)

model, cfg = init_model()


def base64ToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


@app.route('/', methods=['POST'])
def hello():
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

    img_i = random.randint(1, 100000)
    image_path = "./upload_imgs/tmp_{}.jpg".format(img_i)
    cv2.imwrite(image_path, image)
    result = inference_detector(cfg, model, image_path)
    data = format_return_data(result)
    os.remove(image_path)

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

        # We obtain the car 3D information here

        t_pred_x, t_pred_y, t_pred_z = projective_distance_estimation_AAE(json, image_path, camera_matrix, ZRENDER, SCALE)
        json['translation'] = [t_pred_x, t_pred_y, t_pred_z]

    else:
        json = dict(
            status=1,
            msg='NO CAR'
        )
    return jsonify(json)
