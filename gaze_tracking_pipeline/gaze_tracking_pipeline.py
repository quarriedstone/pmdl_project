import argparse
import itertools
import cv2
import dlib
import numpy as np

import face
import time
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
    p1 = tuple(face1.left_eye.eye_corners[1] + face1.left_eye.eye_origin)
    p2 = tuple(face1.left_eye.eye_corners[0] + face1.left_eye.eye_origin)
    print(face1.left_eye.relative_center)
    return frame, face1


def video_demo(save_to, show):
    cap = cv2.VideoCapture(0)

    for i in itertools.count(start=0, step=1):
        _, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        try:
            t = time.time()
            frame, _ = process_frame(frame)
            # print('time: ', time.time()-t)
        except:
            continue

        if save_to:
            cv2.imwrite(save_to + "-{}.png".format(i), frame)
        if show:
            cv2.imshow("Face", frame)


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
def main():
    args = get_args()
    print(args)

    if args.input_photo or args.take_photo:
        photo_demo(photo_filename=args.input_photo, save_to=args.output_prefix, show=args.output_cv2)

    if args.take_video:
        video_demo(save_to=args.output_prefix, show=args.output_cv2)


if __name__ == '__main__':
    main()


def gaze_api(frame):
    frame, face = process_frame(frame)
    return face, frame
