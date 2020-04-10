import numpy as np

# from ref.ellipse_fitting import compute_guaranteedellipse_estimates

MAX_ITER = 200


class structure:
    def __init__(self, t, data_points):
        self.use_pseudoinverse = False
        self.theta_updated = False
        self.lamb = 0.01
        self.k = 0
        self.damping_multiplier = 1.2
        self.gamma = 0.00005
        self.data_points = data_points.transpose()
        self.numberOfPoints = data_points.shape[1]
        f = np.zeros((6, 6))
        f[:3, :3] = [[0, 0, 2],
                     [0, -1, 0],
                     [2, 0, 0]]
        self.F = f
        self.I = np.eye(6, 6)
        self.alpha = 1e-3
        self.tolDelta = 1e-7
        self.tolCost = 1e-7
        self.tolTheta = 1e-7
        self.cost = np.zeros((1, MAX_ITER))
        self.t = np.zeros((6, MAX_ITER))
        self.delta = np.zeros((6, MAX_ITER))
        t = t / np.linalg.norm(t)
        self.t[:, self.k] = t
        self.delta[:, self.k] = np.ones(6)
        self.r = None
        self.jacobian_matrix = None
        self.jacobian_matrix_barrier = None
        self.H = None


# various variable initialisations

def levenbergMarquardtStep(struct):
    jacobian_matrix = struct.jacobian_matrix
    jacobian_matrix_barrier = struct.jacobian_matrix_barrier
    r = struct.r
    I = struct.I
    lamb = struct.lamb
    delta = struct.delta[struct.k]
    damping_multiplier = struct.damping_multiplier
    F = struct.F
    I = struct.I
    t = struct.t[:, struct.k]
    current_cost = struct.cost[:, struct.k][0]
    alpha = struct.alpha
    data_points = struct.data_points
    numberOfPoints = struct.numberOfPoints
    jacob = np.dot(np.hstack([jacobian_matrix.transpose(), jacobian_matrix_barrier]), r)
    tFt = np.linalg.multi_dot([t.transpose(), F, t])
    
    Z_a = np.vstack([np.hstack([(np.dot(jacobian_matrix.transpose(), jacobian_matrix) + lamb * I),
                                tFt ** 4 * np.dot(jacobian_matrix_barrier, jacobian_matrix_barrier.transpose())]),
                     np.hstack([I, -np.dot((tFt) ** 4, I)])])
    zz_a = - np.vstack([jacob, np.zeros((6, 1))])
    update_a = np.linalg.lstsq(Z_a, zz_a)
    update_a = np.array(update_a[0][0:6])
    
    Z_b = np.vstack(
        [np.hstack([(np.dot(jacobian_matrix.transpose(), jacobian_matrix) + (lamb / damping_multiplier) * I),
                    tFt ** 4 * np.dot(jacobian_matrix_barrier, jacobian_matrix_barrier.transpose())]),
         np.hstack([I, -np.dot((tFt) ** 4, I)])])
    zz_b = - np.vstack([jacob, np.zeros((6, 1))])
    update_b = np.linalg.lstsq(Z_b, zz_b)
    update_b = np.array(update_b[0][0:6])
    
    t_potential_a = t.reshape(6, 1) + update_a
    t_potential_b = t.reshape(6, 1) + update_b
    cost_a = 0
    cost_b = 0
    for i in range(numberOfPoints):
        m = data_points[i, :]
        # transformed data point
        ux_j = np.array([m[0] ** 2, m[0] * m[1], m[1] ** 2, m[0], m[1], 1]).reshape(6, 1)
        # derivative of transformed data point
        dux_j = np.array([[2 * m[0], m[1], 0, 1, 0, 0], [0, m[0], 2 * m[1], 0, 1, 0]]).transpose()
        
        # outer  product
        A = np.dot(ux_j, ux_j.transpose())
        
        B = np.dot(dux_j, dux_j.transpose())
        
        t_aBt_a = np.linalg.multi_dot([t_potential_a.transpose(), B, t_potential_a])
        t_aAt_a = np.linalg.multi_dot([t_potential_a.transpose(), A, t_potential_a])
        
        t_bBt_b = np.linalg.multi_dot([t_potential_b.transpose(), B, t_potential_b])
        t_bAt_b = np.linalg.multi_dot([t_potential_b.transpose(), A, t_potential_b])
        
        # AML cost for i'th data point
        cost_a = cost_a + t_aAt_a / t_aBt_a
        cost_b = cost_b + t_bAt_b / t_bBt_b
    
    t_aIt_a = np.linalg.multi_dot([t_potential_a.transpose() , I , t_potential_a])
    t_aFt_a = np.linalg.multi_dot([t_potential_a.transpose() , F , t_potential_a])
    t_bIt_b = np.linalg.multi_dot([t_potential_b.transpose() , I , t_potential_b])
    t_bFt_b = np.linalg.multi_dot([t_potential_b.transpose() , F , t_potential_b])
    
    cost_a = (cost_a + (alpha * (t_aIt_a / t_aFt_a)) ** 2)[0][0]
    cost_b = (cost_b + (alpha * (t_bIt_b / t_bFt_b)) ** 2)[0][0]
    
    # determine  appropriate damping and if possible select an update
    if cost_a >= current_cost and cost_b >= current_cost:
        # neither update reduced the cost
        struct.theta_updated = False
        # no change in the cost
        struct.cost[struct.k + 1] = current_cost
        # no change in parameters
        struct.t[:, struct.k + 1] = t
        # no changes in step direction
        struct.delta[:, struct.k + 1] = delta
        # next iteration add more Identity matrix
        struct.lamb = lamb * damping_multiplier
    elif cost_b < current_cost:
        # update 'b' reduced the cost function
        struct.theta_updated = True
        # store the new cost
        struct.cost[:, struct.k + 1] = cost_b
        a = struct.t[:, struct.k + 1]
        # choose update 'b'
        struct.t[:, struct.k + 1] = (t_potential_b / np.linalg.norm(t_potential_b))[:,0]
        # store the step direction
        struct.delta[:, struct.k + 1] = update_b.transpose()
        # next  iteration add less Identity matrix
        struct.lamb = lamb / damping_multiplier
    else:
        # update 'a' reduced the cost function
        struct.theta_updated = True
        # store the new cost
        struct.cost[struct.k + 1] = cost_a
        # choose update 'a'
        struct.t[:, struct.k + 1] = (t_potential_a / np.linalg.norm(t_potential_a))[:,0]
        # store the step direction
        struct.delta[:, struct.k + 1] = update_a.transpose()
        # keep the same damping for the next iteration
        struct.lamb = lamb
    return struct


def lineSearchStep(struct):
    jacobian_matrix = struct.jacobian_matrix
    jacobian_matrix_barrier = struct.jacobian_matrix_barrier
    r = struct.r
    I = struct.I
    lamb = struct.lamb
    delta = struct.delta[struct.k]
    damping_multiplier = struct.damping_multiplier
    F = struct.F
    I = struct.I
    t = struct.t[:, struct.k]
    current_cost = struct.cost[:,struct.k][0]
    alpha = struct.alpha
    gamma = struct.gamma
    tolDelta = struct.tolDelta
    data_points = struct.data_points
    numberOfPoints = struct.numberOfPoints
    jacob = np.dot(np.hstack([jacobian_matrix.transpose(), jacobian_matrix_barrier]), r)
    tFt = np.linalg.multi_dot([t.transpose(), F, t])

    Z = np.vstack([np.hstack([(np.dot(jacobian_matrix.transpose(), jacobian_matrix) + lamb * I),
                                tFt ** 4 * np.dot(jacobian_matrix_barrier, jacobian_matrix_barrier.transpose())]),
                     np.hstack([I, -np.dot((tFt) ** 4, I)])])
    zz = - np.vstack([jacob, np.zeros((6, 1))])
    update = np.linalg.pinv(Z, 1e-20) * zz
    update = update[1:6]
    frac = 0.5
    while True:
        t_potential = t + frac * update
        delta = frac * update
        frac = frac / 2
        cost = 0
        for i in range(numberOfPoints):
            m = data_points[i, :]
            # transformed data point
            ux_j = np.array([m[0] ** 2, m[0] * m[1], m[1] ** 2, m[0], m[1], 1]).reshape(6, 1)
            # derivative of transformed data point
            dux_j = np.array([[2 * m[0], m[1], 0, 1, 0, 0], [0, m[0], 2 * m[1], 0, 1, 0]]).transpose()
            
            # outer  product
            A = np.dot(ux_j, ux_j.transpose())
            B = np.dot(dux_j, dux_j.transpose())

            tBt = np.linalg.multi_dot([t_potential.transpose(), B, t_potential])
            tAt = np.linalg.multi_dot([t_potential.transpose(), A, t_potential])

            cost = cost + tAt / tBt

        tIt = np.linalg.multi_dot([t_potential.transpose(), I, t_potential])
        tFt = np.linalg.multi_dot([t_potential.transpose(), F, t_potential])

        cost = (cost + (alpha * (tIt / tFt)) ** 2)[0][0]
        if (np.linalg.multi_dot([t_potential.transpose() , F , t_potential]) > 0 and (
                cost < (1 - frac * gamma) * current_cost) or np.linalg.norm(delta) < tolDelta):
            break
    struct.theta_update = True
    struct.t[:, struct.k + 1] = t_potential / np.linalg.norm(t_potential)
    # store the step direction
    struct.delta[:, struct.k + 1] = delta
    # keep the same damping for the next iteration
    struct.cost[:, struct.k + 1] = cost
    return struct


def guaranteed_ellipse_fit(t, data_points):
    keep_going = True
    struct = structure(t, data_points)
    while (keep_going and struct.k < MAX_ITER):
        struct.r = np.zeros((struct.numberOfPoints + 1, 1))
        struct.jacobian_matrix = np.zeros((struct.numberOfPoints, 6))
        struct.jacobian_matrix_barrier = np.zeros((1, 6))
        t = np.array(struct.t[:, struct.k]).reshape(6, 1)
        for i, point in enumerate(struct.data_points):
            ux_j = np.array([point[0] ** 2, point[0] * point[1], point[1] ** 2, point[0], point[1], 1]).reshape(6, 1)
            dux_j = np.array([[2 * point[0], point[1], 0, 1, 0, 0],
                              [0, point[0], 2 * point[1], 0, 1, 0]]).transpose()
            A = np.dot(ux_j, ux_j.transpose())
            B = np.dot(dux_j, dux_j.transpose())
            tBt = np.linalg.multi_dot([t.transpose(), B, t])[0][0]
            tAt = np.linalg.multi_dot([t.transpose(), A, t])[0][0]
            struct.r[i] = np.sqrt(tAt / tBt)
            M = (A / tBt)
            Xbits = B * ((tAt) / (tBt ** 2))
            X = M - Xbits
            grad = ((np.dot(X, t)) / np.sqrt((tAt / tBt))).squeeze()
            struct.jacobian_matrix[i, :] = grad
        tIt = np.linalg.multi_dot([t.transpose(), struct.I, t])
        tFt = np.linalg.multi_dot([t.transpose(), struct.F, t])
        struct.r[-1] = struct.alpha * (tIt / tFt)
        N = (struct.I / tFt)
        Ybits = struct.F * ((tIt) / (tFt) ** 2)
        Y = N - Ybits
        grad_penalty = 2 * struct.alpha * np.dot(Y, t)
        struct.jacobian_matrix_barrier = grad_penalty
        struct.jacobian_matrix_full = np.vstack([struct.jacobian_matrix, struct.jacobian_matrix_barrier.transpose()])
        struct.H = np.dot(struct.jacobian_matrix_full.transpose(), struct.jacobian_matrix_full)
        struct.cost[:,struct.k] = np.dot(struct.r.transpose(), struct.r)
        
        if (not struct.use_pseudoinverse):
            struct = levenbergMarquardtStep(struct)
        else:
            struct = lineSearchStep(struct)
        if (np.linalg.multi_dot([struct.t[:, struct.k + 1].transpose() , struct.F , struct.t[:, struct.k + 1]]) <= 0):
            # from now onwards we will only use lineSearchStep to ensure
            # that we do not overshoot the barrier
            struct.use_pseudoinverse = True
            struct.lamb = 0
            struct.t[:, struct.k + 1] = struct.t[:, struct.k]
            if (struct.k > 1):
                struct.t[:, struct.k] = struct.t[:, struct.k - 1]
        elif (min(np.linalg.norm(struct.t[:, struct.k + 1] - struct.t[:, struct.k]), np.linalg.norm(
                struct.t[:, struct.k + 1] + struct.t[:, struct.k])) < struct.tolTheta and struct.theta_updated):
            keep_going = False
        elif (abs(struct.cost[:,struct.k] - struct.cost[:,struct.k + 1]) < struct.tolCost and struct.theta_updated):
            keep_going = False
        elif (np.linalg.norm(struct.delta[:, struct.k + 1]) < struct.tolDelta and struct.theta_updated):
            keep_going = False
        
        struct.k = struct.k + 1
    theta = struct.t[:, struct.k]
    theta = theta / np.linalg.norm(theta)
    return theta

# guaranteed_ellipse(theta, data_points)
