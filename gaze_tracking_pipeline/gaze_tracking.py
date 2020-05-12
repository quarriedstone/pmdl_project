import argparse
import itertools
import cv2
import dlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
# from train_models import train_xgb
import face
import time

# from train_models import baseline_model

kernel = np.ones((3, 3), np.uint8)
detector = dlib.get_frontal_face_detector()


def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (800, 450))
    faces = detector(frame)
    face1 = face.Face(faces[0], frame)

    frame = face1.frame
    eye_center = (face1.left_eye.center / face1.left_eye.scale + face1.left_eye.eye_origin)
    frame[int(eye_center[1]), int(eye_center[0])] = 255
    eye_center = face1.right_eye.center / face1.right_eye.scale + face1.right_eye.eye_origin
    frame[int(eye_center[1]), int(eye_center[0])] = 255
    return frame, face1


def video_demo(save_to, show):
    cap = cv2.VideoCapture(0)

    for i in itertools.count(start=0, step=1):
        _, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        try:
            t = time.time()
            frame, face1 = process_frame(frame)
            frame = face1.left_eye.eye_region
            eye_center = (face1.left_eye.center)
            frame[int(eye_center[1]), int(eye_center[0])] = 255
            # print(eye_center / frame.shape[0])
            frame2 = face1.right_eye.eye_region
            eye_center = (face1.right_eye.center)
            frame2[int(eye_center[1]), int(eye_center[0])] = 255
            # print(eye_center / frame2.shape[0])
            # frame[int(eye_corner[1]), int(eye_corner[0])] = 255
            # eye_center = face1.right_eye.center / face1.right_eye.scale + face1.right_eye.eye_origin

            # print('2',int(eye_center[1]), int(eye_center[0]))

            # frame[int(eye_center[1]), int(eye_center[0])] = 255
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow(f"Face2", frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print('time: ', time.time() - t)
        except:
            continue

        if save_to:
            cv2.imwrite(save_to + "-{}.png".format(i), frame)
        if show:
            print()
            # frame = face.left_eye.eye_region
            #
            # eye_center = (face.left_eye.center)
            #
            # eye_corner = (face.left_eye.eye_corner)
            # # eye_center = (face1.left_eye.center / face1.left_eye.scale + face1.left_eye.eye_origin)
            # # print('1',int(eye_center[1]), int(eye_center[0]))
            # p1 = (0, int(eye_corner[1]))
            # p2 = (100, int(eye_corner[1]))
            # frame[int(eye_center[1]), int(eye_center[0])] = 255
            # cv2.line(frame, p1, p2, (255), 1)
            # cv2.imshow("Face", frame)


def photo_demo(photo_filename, save_to, show):
    if photo_filename:
        frame = cv2.imread(photo_filename)
    else:
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()

    try:
        frame, _ = process_frame(frame)
    except Exception as e:
        print("Exception occured during frame processing :(")
        exit(1)

    if save_to:
        cv2.imwrite(save_to + ".png", frame)

    if show:
        cv2.imshow("Face", frame)


def get_args():
    parser = argparse.ArgumentParser(
        description='Gaze tracking demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # INPUT
    input_params = parser.add_mutually_exclusive_group(required=True)
    input_params.add_argument('--input-photo', metavar='FILE', help='Photo file to use as input')
    input_params.add_argument('--take-photo', action='store_true', help='Take photo using camera 0')
    input_params.add_argument('--take-video', action='store_true', help='Take video using camera 0')

    # OUTPUT
    model_params = parser.add_argument_group('Outputs')
    model_params.add_argument('--output-prefix', type=str, default=None, metavar='PREFIX',
                              help='Optional prefix enables saving to files')
    model_params.add_argument('--output-cv2', action='store_true', help='Enables output using cv2.imshow()')

    return parser.parse_args()


DEBUG = 0
SCALE = 2
# regressor = train_xgb()

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def baseline_model():
	# create model
	model = Sequential()
	# model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
	# 	# model.add(Dense(7,  activation='relu'))
	# 	# model.add(Dense(2, kernel_initializer='normal'))
	a = 'relu'
	model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation=a))
	model.add(Dense(50, activation=a))
	model.add(Dense(10, activation=a))
	model.add(Dense(2, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['mse'])
	return model
regressor = KerasRegressor(baseline_model())

# This is where you load the actual saved model into new variable.
regressor.model = load_model('saved_model6.h5')
# regressor.model = load_model('saved_model6-back1.h5')
# regressor = train_xgb()


def main():
    args = get_args()
    print(args)

    if args.input_photo or args.take_photo:
        photo_demo(photo_filename=args.input_photo, save_to=args.output_prefix, show=args.output_cv2)

    if args.take_video:
        video_demo(save_to=args.output_prefix, show=args.output_cv2)


if __name__ == '__main__':
    main()

def predict(frame):
    vector = gaze_api(frame)
    result = regressor.predict(vector)
    return result

def gaze_api(frame):
    frame, face = process_frame(frame)
    vector = np.expand_dims(np.concatenate((face.left_eye.center_coefs, face.right_eye.center_coefs, np.array([face.eye_y]).reshape((1)),
                             face.face_position[0].squeeze(),
                             face.face_position[1].squeeze())), 0)

    # result = regressor.predict(vector)
    return vector

