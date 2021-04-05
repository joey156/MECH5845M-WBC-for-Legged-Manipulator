import numpy as np
from qpoases import PyQProblemB as QProblemB
from qpoases import PyBooleanType as BooleanType
from qpoases import PySubjectToStatus as SubjectToStatus
from qpoases import PyOptions as Options

class QP:
    def __init__(self, A, b, lb, ub):

        self.A = A
        self.b = b
        self.lb = lb
        self.ub = ub
        self.H = np.dot(A.T, A)
        self.g = np.dot(b.T, A)
        self.no_solutions = 26
        self.nWSR = np.array([250])

    def solveQP(self):
        #initialise qp
        qp = QProblemB(self.no_solutions)
        
        # set up options
        options = Options()
        options.enableFlippingBounds = BooleanType.FALSE
        options.initialStatusBounds = SubjectToStatus.INACTIVE
        options.numRefinementSteps = 1

        qp.setOptions(options)
        
        #solve qp
        qp.init(self.H, self.g, self.lb, self.ub, self.nWSR)

        self.xOpt = np.zeros((self.no_solutions,))
        qp.getPrimalSolution(self.xOpt)
        return self.xOpt
    