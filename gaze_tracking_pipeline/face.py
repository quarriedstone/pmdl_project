import os
import urllib
import cv2
import dlib
import numpy as np

from eye import Eye

if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
	print("Downloading face landmarks model...")
	urllib.request.urlretrieve(
		"https://raw.githubusercontent.com/AKSHAYUBHAT/TensorFace/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat",
		"shape_predictor_68_face_landmarks.dat")

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


class Face:
	def __init__(self, face, frame):
		self.face = face
		self.frame = frame
		self.landmarks = predictor(frame, face)
		self.x_ref_point = None
		self.face_position = self.posit_predict()

		eye1 = np.array([[self.landmarks.part(mark).x, self.landmarks.part(mark).y] for mark in range(36, 42)])
		eye2 = np.array([[self.landmarks.part(mark).x, self.landmarks.part(mark).y] for mark in range(42, 48)])
		self.left_eye = Eye(eye1, self.frame, 'left')

		self.right_eye = Eye(eye2, self.frame, 'right')
		d = np.linalg.norm(np.cross(self.right_eye.global_center - self.left_eye.global_center,
		                            self.left_eye.global_center - self.central_line)) / np.linalg.norm(
			self.right_eye.global_center - self.left_eye.global_center)
		n = np.linalg.norm(np.cross(self.right_eye.global_center - self.left_eye.global_center,
		                            self.left_eye.global_center - self.chin)) / np.linalg.norm(
			self.right_eye.global_center - self.left_eye.global_center)
		self.eye_y = d

	def posit_predict(self):
		landmarks = self.landmarks
		between_eyes = (landmarks.part(27).x, landmarks.part(27).y)
		chin = (landmarks.part(8).x, landmarks.part(8).y)
		tip = (landmarks.part(30).x, landmarks.part(30).y)
		lipL = (landmarks.part(54).x, landmarks.part(54).y)
		lipR = (landmarks.part(48).x, landmarks.part(48).y)
		eyeL = (landmarks.part(45).x, landmarks.part(45).y)
		eyeR = (landmarks.part(36).x, landmarks.part(36).y)
		image_points = np.array([tip, chin, eyeL, eyeR, lipL, lipR], dtype="double")

		# self.x_ref_point = between_eyes + (chin - between_eyes)/2

		model_points = np.array([
			(0.0, 0.0, 0.0),  # Nose tip
			(0.0, -330.0, -65.0),  # Chin
			(-225.0, 170.0, -135.0),  # Left eye left corner
			(225.0, 170.0, -135.0),  # Right eye right corner
			(-150.0, -150.0, -125.0),  # Left Mouth corner
			(150.0, -150.0, -125.0)  # Right mouth corner

		])

		size = self.frame.shape
		focal_length = size[1]
		center = (size[1] / 2, size[0] / 2)

		camera_matrix = np.array(
			[[focal_length, 0, center[0]],
			 [0, focal_length, center[1]],
			 [0, 0, 1]], dtype="double"
		)

		dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
		(success, rotation_vector, translation_vector) = cv2.solvePnP(
			model_points,
			image_points,
			camera_matrix,
			dist_coeffs
		)

		# Project a 3D point (0, 0, 1000.0) onto the image plane.
		# We use this to draw a line sticking out of the nose
		(nose_end_point2D, jacobian) = cv2.projectPoints(
			np.array([(0.0, 0.0, 500.0)]),
			rotation_vector,
			translation_vector,
			camera_matrix,
			dist_coeffs
		)

		# for p in image_points:
		#     cv2.circle(self.frame, (int(p[0]), int(p[1])), 3, (255), -1)

		p1 = (int(image_points[0][0]), int(image_points[0][1]))
		p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
		# print(translation_vector)
		cv2.line(self.frame, p1, p2, (255), 2)

		chin1 = np.array([landmarks.part(8).x, landmarks.part(8).y])
		self.central_line = between_eyes + (between_eyes - chin1) / 2
		self.chin = chin1

		return rotation_vector, translation_vector
