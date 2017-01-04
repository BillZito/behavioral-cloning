import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

#set up socketio connection (to send info to driving simulator?)
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

#telemetry isn't set anywhere--assuming this is called by simulator
#and therefore sid and data are based in by processor
@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # print('initial steering angle is', steering_angle)
    # The current throttle of the car
    throttle = data["throttle"]
    # print('initial throttle is', throttle)
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array = np.array([image_array])
    image_array = np.moveaxis(image_array, 3, 1)

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    #changed from transformed image array since I do it myself... does our model predict doing the same things we defined?
    steering_angle = float(model.predict(image_array, batch_size=1))

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = .1
    print('new steering angle is', steering_angle)
    # print('new throttle is', throttle)
    send_control(steering_angle, throttle)

#making a connection
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

#send control sets the steering angle and throttle based on start
def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    #finding model path
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    #opening the model with read-- this should work for my model
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    #compiling model.. i shouldnt need to do that again right?
    model.compile("adam", "mse")
    #weights file doesnt exist yet... google this
    weights_file = args.model.replace('json', 'h5')
    #load weights into model
    model.load_weights(weights_file)
    # print('printing summary now ')
    # model.summary()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)