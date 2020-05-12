import math
import cv2
import numpy as np
import operator
from scipy import optimize
from functools import reduce
import scipy.signal
from iris import Iris

DEBUG = 0
SCALE = 2
def order_points(coords):
	center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
	return (sorted(coords, key=lambda coord: (-135 - math.degrees(
		math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))


def chaikins_corner_cutting(coords, refinements=3):
	coords = np.array(coords)
	for _ in range(refinements):
		L = coords.repeat(2, axis=0)
		R = np.empty_like(L)
		R[0] = L[0]
		R[2::2] = L[1:-1:2]
		R[1:-1:2] = L[2::2]
		R[-1] = L[-1]
		coords = L * 0.75 + R * 0.25
	return coords



class Eye:
	def __init__(self, landmarks, frame, side):
		self.side = side
		self.frame = frame
		self.scale = 1
		self.eye_points = self._extract_eye_points(landmarks)
		self.bbox = cv2.boundingRect(self.eye_points)
		self.eye_region, self.eye_origin = self._extract_eye_region()

		self.mask = self._make_mask()
		left, right = self._generate_masked_gradient()
		self.iris = Iris(left, right, self.eye_region, self.mask)
		self.eye_corner, self.eye_width = self._extract_eye_corners()
		self.center, self.r = self._calculate_center()
		self.global_center = self.eye_origin + self.center / self.scale
		self.center_coefs = self._calc_i_position()

	# def _calculate_relative_center(self):
	# 	corner = np.array(self.eye_corner)
	# 	result = (corner - self.center)
	# 	result[0] = result[0]/self.eye_width
	# 	return result

	def _calculate_center(self):
		global R
		candidate_points = []
		if len(self.iris.left_gradient) > 0:
			candidate_points.extend(self.iris.left_gradient.transpose())
		if len(self.iris.right_gradient) > 0:
			candidate_points.extend(self.iris.right_gradient.transpose())
		candidate_points = np.array(candidate_points)

		def calc_R(xc, yc):
			x = candidate_points[:, 1]
			y = candidate_points[:, 0]
			""" calculate the distance of each 2D points from the center (xc, yc) """
			x = x - xc
			y = y - yc
			return np.power((np.power(x, 2) + np.power(y, 2)), 0.5)

		def f_2(c):
			global R
			""" calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
			Ri = calc_R(*c)
			R = np.mean(Ri)
			# alpha = 1000000 if (R < 27 or R > 37) else 0
			if (R < 27 or R > 37): R = 30
			return Ri - R

		circle = optimize.least_squares(f_2, self.iris.rough_center, ftol=1e-1)
		center = circle['x'].astype(int)
		# print('rad= ', R)
		# print('cent= ', center)
		return np.array(center), R

	def _fix_pupil_center_by_physiology(self, up_point, down_point):
		pupil_to_iris_ratio = 1 / 5
		if self.center[1] - up_point < int(self.r * pupil_to_iris_ratio):
			self.center[1] = up_point + int(self.r * pupil_to_iris_ratio)
		if down_point - self.center[1] < int(self.r * pupil_to_iris_ratio):
			self.center[1] = down_point - int(self.r * pupil_to_iris_ratio)
		if down_point - up_point < pupil_to_iris_ratio * 2:
			self.center[1] = up_point + (down_point - up_point) / 2

	def _calc_i_position(self):
		sobely = cv2.Sobel(self.eye_region, cv2.CV_64F, 0, 1, ksize=3)
		h = sobely.shape[0]
		roi = [int(sobely.shape[1] / 2 - sobely.shape[1] / 5), int(sobely.shape[1] / 2 + sobely.shape[1] / 5)]
		up = sobely[0:int(h / 2), roi[0]:roi[1]] * -1
		down = sobely[int(h / 2):, roi[0]:roi[1]]
		# down = abs(sobely[int(h / 2):, :] * -1)
		up_y_reference = np.argmax(np.sum(up, axis=1))
		landmark_down_ref = h / self.scale
		down_y_reference = (np.sum(down, axis=1))
		down_peak = np.argmax(down_y_reference)
		coef_of_opened = (up.shape[0] + landmark_down_ref - self.center[1]) / self.r
		down_y_reference = down_y_reference * -1
		down_y_reference[np.where(down_y_reference < 0)] = 0
		peaks = scipy.signal.find_peaks(down_y_reference, height=100)
		# peaks = scipy.signal.find_peaks(down_y_reference, width=5, distance=5, height=1000)
		if len(peaks[0]) >= 1 and down_peak < peaks[0][-1] and coef_of_opened > 1:
			down_peak = peaks[0][0]
		# elif len(peaks[0]) > 1:
		# 	potential_peaks = peaks[0][1:]
		# 	potential_peaks_heights = peaks[1]['peak_heights'][1:]
		# 	highest = np.argmax(potential_peaks_heights)
		# 	down_peak = potential_peaks[highest]
		down_y_reference = up.shape[0] + down_peak
		self._fix_pupil_center_by_physiology(up_y_reference, down_y_reference)
		self.eye_region[down_y_reference, :] = 255
		self.eye_region[up_y_reference, :] = 255
		iris_y_component1 = (down_y_reference - self.center[1])
		iris_y_component2 = (self.center[1] - up_y_reference)

		corner = np.array(self.eye_corner)
		iris_x_component = (corner[0] - self.center[0])
		iris_x_component = iris_x_component / self.eye_width

		# print(self.side, iris_y_component1, iris_y_component2)
		# print(self.r)
		# return np.expand_dims(np.array(iris_x_component), 0)
		return iris_x_component, iris_y_component1, iris_y_component2

	def _extract_eye_corners(self):
		corner_l = self.eye_points[np.where(self.eye_points[:, 0] == min(self.eye_points[:, 0]))].squeeze()
		corner_r = self.eye_points[np.where(self.eye_points[:, 0] == max(self.eye_points[:, 0]))].squeeze()
		reg_rad = 2
		# def high_def_corner(point):
		# 	corner_reg = self.frame[point[1]-reg_rad:point[1]+reg_rad, point[0]-reg_rad:point[0]+reg_rad]
		# 	corner_reg = cv2.resize(corner_reg, (int(corner_reg.shape[1] * self.scale), int(corner_reg.shape[0] * self.scale)),
		# 	                    interpolation=cv2.INTER_CUBIC)
		# 	pr
		if self.side == 'right':
			corner = corner_l

		if self.side == 'left':
			corner = corner_r

		coef = corner_r[0] - corner_l[0]
		return corner * self.scale, coef

	def _extract_eye_points(self, landmarks):
		eye_points = chaikins_corner_cutting(landmarks).astype(int)
		eye_points = np.array(order_points(eye_points[np.unique(eye_points[:, 0], return_index=True, axis=0)[1]]))
		return eye_points

	def _make_mask(self):
		mask = np.ones((self.eye_region.shape[0], self.eye_region.shape[1]), np.uint8) * 255
		cv2.fillConvexPoly(mask, (self.eye_points * self.scale).astype(int), 0)
		mask = np.array(mask, dtype=int).astype('uint8')
		mask = 255 - mask
		# mask = cv2.resize(mask, (mask.shape[1]*SCALE, mask.shape[0]*SCALE), interpolation = cv2.INTER_CUBIC)

		return mask

	def _extract_eye_region(self):
		global SCALE
		x, y, w, h = self.bbox
		frame_padding = int(h / 2.)
		frame_padding = 3
		croped = self.frame[y - frame_padding:y + h + frame_padding, x:x + w].copy()
		eye_origin = (x, y - frame_padding)
		self.eye_points = self.eye_points - eye_origin
		if SCALE == 2:
			self.scale = (120 / croped.shape[1])
			croped = cv2.resize(croped, (int(croped.shape[1] * self.scale), int(croped.shape[0] * self.scale)),
			                    interpolation=cv2.INTER_CUBIC)
		# print(croped.shape)
		croped = cv2.bilateralFilter(croped, 15, 60, 60)

		return croped, np.array(eye_origin)

	def _generate_masked_gradient(self):
		sobelx = cv2.Sobel(self.eye_region, cv2.CV_64F, 1, 0, ksize=3)
		sobely = cv2.Sobel(self.eye_region, cv2.CV_64F, 1, 1, ksize=3)
		sobelx = sobelx + sobely
		sobelx = sobelx / np.max(sobelx) * 128
		grad_right = sobelx.copy()
		grad_right[np.where(grad_right < 0)] = 0
		grad_left = sobelx.copy()
		grad_left[np.where(grad_left > 0)] = 0
		grad_left = abs(grad_left)
		grad_left = np.array(grad_left, dtype='uint8')
		grad_right = np.array(grad_right, dtype='uint8')
		# grad_left = cv2.dilate(grad_left, kernel, iterations=1)  # todo JUST ADDED
		# grad_left = cv2.erode(grad_left, kernel, iterations=1)
		# grad_right = cv2.bilateralFilter(grad_right, 15, 30, 30)
		# grad_left = cv2.bilateralFilter(grad_left, 15, 30, 30)
		grad_right = cv2.bitwise_and(grad_right, self.mask)
		grad_left = cv2.bitwise_and(grad_left, self.mask)
		return grad_left, grad_right