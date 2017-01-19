import cv2
import base64
import json
import time
import argparse
import socketio
import eventlet
import numpy as np
import eventlet.wsgi
from PIL import Image
from io import BytesIO
from PIL import ImageOps
import matplotlib.pyplot as plt
from flask import Flask, render_template

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf
from process_data import show_image



sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

steering_angle_ls = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    # print('image array shape', image_array.shape)
    # image_array = image_array.reshape((1,) + image_array.shape)
    # print('transofrmed size', transformed_image_array.shape)
    # image_array = image_array[20:160]
    # preprocessing
    # show_image(image_array)
    resized = cv2.resize(image_array, (200, 66))
    # show_image(resized)

    resized = resized.reshape((1,) + resized.shape)
    # resized = ( cv2.resize((cv2.cvtColor(image_array[0], cv2.COLOR_RGB2HSV))[:,:,1],(32,16))).reshape(1,16,32,1)
    # resized = cv2.resize(cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)[:, :,1], (32, 16))
    # resized = resized.reshape((1,) + resized.shape + (1,))
    # print('resized shape', resized.shape)
    # plt.imshow(resized)
    # plt.show()
    # 
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(resized, batch_size=1))
    # if abs(steering_angle) > .1:
    steering_angle = steering_angle * 1.3
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.1
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


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