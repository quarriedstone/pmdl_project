import math
import cv2
import imutils
import numpy as np
import scipy.ndimage


def find(ret):
    contours = cv2.findContours(
        ret.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = contours[0]
    ((x, y), radius) = cv2.minEnclosingCircle(c)

    return int(x), int(y)


class Iris:
    def __init__(self, left, right, eye_frame, mask):
        self.mask = mask
        self.rough_center = self._compute_rough_center(eye_frame)
        self.shape = left.shape
        self.r = self.shape[1] / 4
        left_border = self.rough_center[0] > self.r
        right_border = self.rough_center[0] < self.shape[1] - 0.6 * self.r  # TODO magic coef
        self.right_gradient = []
        self.left_gradient = []

        approx_r, approx_l = self._approximate_iris_border(
            n=10,
            left_border=left_border,
            right_border=right_border
        )

        if approx_l:
            candidate_region_l = self._iris_border_dilation(approx_l)
            left = self._squeeze_gradient(left)
            left = left * candidate_region_l
            self.left_gradient = np.array(np.where(left > 0))

        if approx_r:
            candidate_region_r = self._iris_border_dilation(approx_r)
            right = self._squeeze_gradient(right)
            right = right * candidate_region_r
            self.right_gradient = np.array(np.where(right > 0))

    def _compute_rough_center(self, eye_frame):
        croped = cv2.equalizeHist(eye_frame)
        croped = cv2.medianBlur(croped, 7)
        croped = croped + self.mask

        croped[np.where(croped > 255)] = 255

        croped = np.array(croped, dtype=float)
        croped = np.power(croped, 3)  # TODO ???
        croped = np.array(croped / croped.min(), dtype=int)

        croped[np.where(croped > 255)] = 255

        croped = 255 - croped
        croped = croped.astype('uint8')
        pupil = find(croped)

        return pupil

    def _squeeze_gradient(self, img):
        img_copy = img.copy()
        non_zero = img_copy[np.where(img_copy > 0)]
        d = np.percentile(non_zero, 60)
        img_copy[np.where(img_copy < d)] = 0
        img_copy[np.where(img_copy > 0)] = 1
        img = img * img_copy
        xyw = []
        [[xyw.append([x, y, img[y, x] ** 1 / 3]) for x in range(img.shape[1]) if img[y, x] > 0]
         for y in range(img.shape[0])]  # FIXME this is bad

        # cluster experiment
        points_for_clustering = []
        img_copy = np.zeros_like(img).astype(float)

        for i, line in enumerate(img):
            non_zero_ind = np.where(line > 0)[0]
            non_zero = line[non_zero_ind]

            if len(non_zero) > 2:
                target = np.vstack([non_zero_ind, non_zero]).transpose()
                target = target[target[:, 1].argsort()]
                target = target[-3:, :]
                for t in target[:, 0]:
                    points_for_clustering.append([i, t])
                    img_copy[i, int(t)] = 1

        kernel = np.ones((2, 2))
        img_copy = cv2.dilate(img_copy, kernel, iterations=1)
        clustered = scipy.ndimage.label(img_copy)[0]
        c = clustered.flatten()
        c = c[np.where(c > 0)]
        counts = np.bincount(c)
        element = np.argmax(counts)
        img_copy = np.zeros_like(img).astype(float)
        img_copy[np.where(clustered == element)] = 1
        img_copy = cv2.erode(img_copy, kernel, iterations=1)

        return img_copy

    def _approximate_iris_border(self, n, left_border, right_border):
        x0, y0 = self.rough_center
        points_l = []
        points_r = []
        delta_fi = math.pi / (2 * n)
        fi0 = math.pi / 12

        for i in range(n):
            xi = int(x0 + self.r * math.cos(fi0 - delta_fi * i))
            xi_2 = int(2 * x0 - xi)
            yi = int(y0 - self.r * math.sin(fi0 - delta_fi * i))
            if left_border: points_l.append([xi, yi])
            if right_border: points_r.append([xi_2, yi])

        return points_l, points_r

    def _iris_border_dilation(self, coords):
        polygon = np.zeros(self.shape, dtype='uint8')

        for point in coords:
            if point[0] < self.shape[1] and point[1] < self.shape[0]:
                polygon[point[1], point[0]] = 255

        kernel = np.ones((3, 3), np.uint8)
        dilated_polygon = cv2.dilate(polygon, kernel, iterations=4) / 255

        return dilated_polygon
