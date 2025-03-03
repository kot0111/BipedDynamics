from transverse_linearization.linearization import TransverseLinearization

class Feedback:
    def __init__(self, tl: TransverseLinearization):
        self.tl = tl
        self.constraints = tl.constraints

    def get_transverse(self, state):
        theta, q2, q3, dtheta, dq2, dq3 = state

        #TODO Ну и так понятно 
        I = 0

        y1 = q2 - self.constraints(theta)[0]
        y2 = q3 - self.constraints(theta)[1]
        dy1 = dq2 - self.constraints(theta)[2] * dtheta
        dy2 = dq3 - self.constraints(theta)[3] * dtheta
        
        return [I, y1, y2, dy1, dy2]
    
    def __call__(self, t, y, state):
        theta, q2, q3, dtheta, dq2, dq3 = state

        #TODO Может стоит пересчитывать номинальное управление по нормальному, а не на идеальной траектории
        # utr = 
        u = self.constraints(theta)[6]

        return u