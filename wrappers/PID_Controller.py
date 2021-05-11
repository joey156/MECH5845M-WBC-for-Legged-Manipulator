import numpy as np
import time

class PID:
    def __init__(self, Kp, ts, C, Ki=0, Kd=0):
        self.Kp = Kp # controller gain
        self.Ki = Ki # intergral time constant
        self.Kd = Kd # derivative time constant
        self.ts = ts # sampling time
        self.C = C # initial controller value
        self.t_prev = time.time() # time - 1
        self.I = np.zeros(7)
        self.e_previous = 0 # last error value#
        print("set val")
        print(self.C)
        

    def PIDUpdate(self, set_point, feedback):
        t = time.time()
        e_current = set_point - feedback
        dt = t - self.t_prev
        de = e_current - self.e_previous

        P = self.Kp * e_current

        if self.Ki != 0:
            self.I = self.I + (self.Ki*e_current*dt)
        else:
            self.I = 0

        D = self.Kd*(de/dt)

        output = P + self.I + D + self.C
        print(P)
        print(self.I)
        print(D)
        
        self.t_prev = t
        self.e_previous = e_current

        return output

        
