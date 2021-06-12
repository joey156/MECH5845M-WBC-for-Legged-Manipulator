import numpy as np
from qpoases import PyQProblemB as QProblemB
from qpoases import PySQProblem as SQProblem
from qpoases import PyBooleanType as BooleanType
from qpoases import PySubjectToStatus as SubjectToStatus
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel

class QP:
    def __init__(self, A, b, lb, ub, C=None, Clb=None, Cub=None, n_of_velocity_dimensions=None):

        self.lb = lb
        self.ub = ub
        self.Clb = Clb
        self.Cub = Cub
        self.C = C
        self.H = np.dot(A.T, A)
        self.g = np.dot(-A.T, b)
        self.no_solutions = n_of_velocity_dimensions
        self.nWSR = np.array([100000])
        self.qp = None

    def solveQP(self):
        #initialise qp
        if self.C is None or self.Clb is None or self.Cub is None:
            self.qp = QProblemB(self.no_solutions)

        else:
            self.qp = SQProblem(self.no_solutions, self.C.shape[0])
        
        # set up options
        options = Options()
        options.setToFast()
        options.printLevel = PrintLevel.NONE
        options.enableFlippingBounds = BooleanType.FALSE
        #options.initialStatusBounds = SubjectToStatus.INACTIVE
        options.numRefinementSteps = 100

        self.qp.setOptions(options)

        
        
        #solve qp
        if self.C is None or self.Clb is None or self.Cub is None:
            self.qp.init(self.H, self.g, self.lb, self.ub, self.nWSR)

        else:
            self.qp.init(self.H, self.g, self.C, self.lb, self.ub, self.Clb, self.Cub, self.nWSR)

        self.xOpt = np.zeros((self.no_solutions,))
        self.qp.getPrimalSolution(self.xOpt)

        return self.xOpt

    def solveQPHotstart(self, A, b, lb, ub, C, Clb, Cub):

        if self.Clb is None or self.Cub is None:
            print("Error, cannot hotstart simply bounded QP")
            exit()

        self.lb = lb
        self.ub = ub
        self.Clb = Clb
        self.Cub = Cub
        self.C = C
        self.H = np.dot(A.T, A)
        self.g = np.dot(-A.T, b)
        self.nWSR = np.array([100000])

        self.qp.hotstart(self.H, self.g, self.C, self.lb, self.ub, self.Clb, self.Cub, self.nWSR)
        self.qp.getPrimalSolution(self.xOpt)

        return self.xOpt
    
