import cv2
import json
import time
import base64
import argparse
import socketio
import eventlet
import numpy as np
import eventlet.wsgi
from PIL import Image
from io import BytesIO
from PIL import ImageOps
from flask import Flask, render_template
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

from process_data import show_image, show_images

#set up socketio connection (to send info to driving simulator?)
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

#telemetry isn't set anywhere--assuming this is called by simulator
#and therefore sid and data are based in by processor
@sio.on('telemetry')
def telemetry(sid, data):
    print('tel called')
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    # print('data is', data)
    # center_name = img_name.replace('left', 'center')
    # right_name = img_name.replace('left', 'right')

    # left_img = misc.imread(src_dir + '/' + img_name)
    # center_img = misc.imread(src_dir + '/' + center_name)
    # right_img = misc.imread(src_dir + '/' + right_name)

    # img = np.concatenate((left_img, center_img, right_img), axis=1)

    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    # show_image(image_array)
    image_array = image_array[60:140]
    # show_image(image_array)
    # resize to 64, 64 and put in shape [1, 64, 64, 3] for model prediction
    image_array = cv2.resize(image_array, (40, 160))
    # show_image(image_array)
    image_array = np.array([image_array])
    # show_images(image_array)

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(image_array, batch_size=1))
    if abs(steering_angle) > .1:
        steering_angle = steering_angle * 1.2
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = .08
    print('new steering angle is', steering_angle)
    send_control(steering_angle, throttle)

# can change throttle and add additional images if we want....

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