import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd
        self.last_u = 0

        A = np.eye(3,k=1)
        B = np.array([[0], [self.b], [0]])
        L = np.array([[3*p], [3*p**2], [p**3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)
        

    def set_b(self, b):
        self.eso.set_B(np.array([[0], [b], [0]]))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q, q_dot = x
        q_est, q_est_dot, f = self.eso.get_state()
        self.eso.update(q, self.last_u)
        v = self.kp * (q_d - q) + self.kd * (q_d_dot - q_est_dot) + q_d_ddot
        u = (v - f) / self.b
        self.last_u = u
        return u
