import math
import cv2
import numpy as np
import operator
from scipy import optimize
from functools import reduce

from iris import Iris


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


def _extract_eye_points(landmarks):
    eye_points = chaikins_corner_cutting(landmarks).astype(int)
    eye_points = np.array(order_points(eye_points[np.unique(eye_points[:, 0], return_index=True, axis=0)[1]]))
    return eye_points


class Eye:
    def __init__(self, landmarks, frame):
        self.frame = frame
        self.eye_points = _extract_eye_points(landmarks)
        self.bbox = cv2.boundingRect(self.eye_points)
        self.eye_region, self.eye_origin = self._extract_eye_region(frame_padding=2)
        self.mask = self._make_mask()
        left, right = self._generate_masked_gradient()
        self.iris = Iris(left, right, self.eye_region, self.mask)
        self.eye_corners = self._extract_eye_corners()
        self.center = self._calculate_center()

    def _calculate_center(self):
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
            return Ri - R

        circle = optimize.least_squares(f_2, self.iris.rough_center, ftol=1e-4)
        center = circle['x'].astype(int)

        return center

    def _extract_eye_corners(self):
        left_corner = self.eye_points[np.where(self.eye_points == min(self.eye_points[:, 0]))]
        right_corner = self.eye_points[np.where(self.eye_points == max(self.eye_points[:, 0]))]

        return left_corner, right_corner

    def _make_mask(self):
        mask = np.ones((self.eye_region.shape[0], self.eye_region.shape[1]), np.uint8) * 255
        cv2.fillConvexPoly(mask, self.eye_points, 0)
        mask = np.array(mask, dtype=int).astype('uint8')
        mask = 255 - mask
        return mask

    def _extract_eye_region(self, frame_padding):
        x, y, w, h = self.bbox
        croped = self.frame[y - frame_padding:y + h + frame_padding, x:x + w].copy()
        eye_origin = (x, y - frame_padding)
        self.eye_points = self.eye_points - eye_origin
        croped = cv2.bilateralFilter(croped, 15, 30, 30)

        return croped, eye_origin

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
        grad_right = cv2.bitwise_and(grad_right, self.mask)
        grad_left = cv2.bitwise_and(grad_left, self.mask)

        return grad_left, grad_right
