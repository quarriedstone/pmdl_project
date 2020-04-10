import numpy as np
from guaranteed_ellipse import guaranteed_ellipse_fit


def compute_guaranteedellipse_estimates(data_points):
	assert data_points.shape[0] == 2
	data_points = np.vstack([data_points, np.ones(data_points.shape[1])])
	normalized_points, T = normalise2dpts(data_points)
	# theta = direct_ellipse_fit(data_points)
	# theta = direct_ellipse_fit(data_points)
	# print(get_ellipse_center(theta))
	theta = guaranteed_ellipse_fit(direct_ellipse_fit(normalized_points), normalized_points)
	theta / np.linalg.norm(theta)

	return get_ellipse_center(theta, T=T)


# return normalized_points, theta
# theta = guaranteedEllipseFit(theta, data_points)

def normalise2dpts(pts):
	assert pts.shape[0] == 3
	assert not np.isinf(pts).any()
	c = np.mean(pts, axis=1).reshape(3, 1)
	new_pts = np.subtract(pts, c)
	dist = np.sqrt(new_pts[0] ** 2 + new_pts[1] ** 2)
	meandist = np.mean(dist)
	scale = np.sqrt(2) / meandist
	T = np.array([[scale, 0, - scale * c[0][0]],
				  [0, scale, - scale * c[1][0]],
				  [0, 0, 1]])
	result_points = np.dot(T, pts)
	return result_points, T


def direct_ellipse_fit(data):  # x-0, y-1
	x = data[0, :]
	y = data[1, :]
	D1 = np.array([x ** 2, x * y, y ** 2]).transpose()  # quadratic part of the design matrix
	D2 = np.array([x, y, np.ones(len(x))]).transpose()  # linear part of the design matrix
	S1 = np.dot(D1.transpose(), D1)  # quadratic part of the scatter matrix
	S2 = np.dot(D1.transpose(), D2)  # combined part of the scatter matrix
	S3 = np.dot(D2.transpose(), D2)  # linear part of the scatter matrix
	T = - np.dot(np.linalg.inv(S3), S2.transpose())  # for getting a2 from a1
	M = S1 + np.dot(S2, T)  # reduce scatter matrix
	M = np.array([M[2] / 2, - M[1], M[0] / 2])  # premultiply by inv(C1)
	[evalue, evec] = np.linalg.eig(M)  # solve eigensystem
	evalue = np.diag(evalue)
	cond = 4 * evec[0] * evec[2] - evec[1] ** 2  # evaluate a'Ca
	al = np.squeeze(evec[:, np.where(cond > 0)])  # eigenvector for min. pos. eigenvalue
	a = np.hstack([al, np.dot(T, al)])  # ellipse coefficients
	a = a / np.linalg.norm(a)
	return a


def get_ellipse_center(theta, T=[]):
	A, B, C, D, E, F = theta
	if len(T) > 0:
		C = [[A, B / 2, D / 2],
			 [B / 2, C, E / 2],
			 [D / 2, E / 2, F]]
		C = np.linalg.multi_dot([T.transpose(), C, T])
		a = C[0,0]
		b = C[0,1] * 2
		d = C[0,2] * 2
		c = C[1,1]
		e = C[1,2] * 2
		f = C[2,2]
		A, B, C, D, E, F = [a,b,c,d,e,f]
		
	coeff = [(B ** 2 - 4 * A * C), 2 * (B * E - 2 * C * D), E ** 2 - 4 * C * F]
	x_extemes = np.roots(coeff)
	
	def calc_ellipse_y(x):
		coeff = [C, B * x + E, A * x ** 2 + D * x + F]
		return np.roots(coeff)
	
	y1, y2 = [calc_ellipse_y(x) for x in x_extemes]
	
	def ellipse_center(x1, y1, x2, y2):
		x_c = x1 + (x2 - x1) / 2
		y_c = y1 + (y2 - y1) / 2
		return int(x_c), int(y_c[0])
	
	center = ellipse_center(x_extemes[0], y1, x_extemes[1], y2)
	return center


def guaranteedEllipseFit(theta, x):
	pass


data_points = [[85, 54],
			   [85, 58],
			   [45, 54],
			   [42, 40],
			   [77, 64],
			   [82, 61],
			   [44, 50],
			   [41, 43],
			   [44, 58],
			   [47, 61],
			   [41, 47],
			   [89, 51],
			   [79, 62],
			   [87, 47],
			   [88, 40],
			   [85, 37],
			   [53, 64],
			   [89, 43]]

data_points = np.array(data_points).transpose()
theta = compute_guaranteedellipse_estimates(data_points)
