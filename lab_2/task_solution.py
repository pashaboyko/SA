# -*- coding: utf-8 -*-

from copy import deepcopy
from scipy import special
from openpyxl import Workbook
import numpy as np
from lab_2.optimization_methods import *
from tabulate import tabulate as tb


class Solve(object):

    def __init__(self,d):
        self.n = d['samples']
        self.deg = d['dimensions']
        self.filename_input = d['input_file']
        self.filename_output = d['output_file']
        self.dict = d['output_file']
        self.p = list(map(lambda x:x+1,d['degrees'])) # on 1 more because include 0
        self.weights = d['weights']
        self.poly_type = d['poly_type']
        self.splitted_lambdas = d['lambda_multiblock']
        self.solving_method = d['method']
        self.eps = 1E-6
        self.norm_error=0.0
        self.error=0.0


    def define_data(self):
        f = open(self.filename_input, 'r')
        # all data from file_input in float
        self.datas = np.matrix([list( map( lambda x:float(x),f.readline().split() ) ) for i in range(self.n)])
        # list of sum degrees [ 3,1,2] -> [3,4,6]
        self.degf = [sum(self.deg[:i + 1]) for i in range(len(self.deg))]

    def _minimize_equation(self, A, b):
        """
        Finds such vector x that |Ax-b|->min.
        :param A: Matrix A
        :param b: Vector b
        :return: Vector x
        """
        if self.solving_method == 'LSTM':
            return np.linalg.lstsq(A,b)[0]
        elif self.solving_method == 'conjucate':
            return conjugate_gradient_method(A.T*A, A.T*b, self.eps)
        elif self.solving_method == 'coordDesc':
            return coordinate_descent(A.T*A, A.T*b, self.eps)

    def norm_data(self):
        '''
        norm vectors value to value in [0,1]
        :return: float number in [0,1]
        '''
        n,m = self.datas.shape
        vec = np.ndarray(shape=(n,m),dtype=float)
        for j in range(m):
            minv = np.min(self.datas[:,j])
            maxv = np.max(self.datas[:,j])
            for i in range(n):
                vec[i,j] = (self.datas[i,j] - minv)/(maxv - minv)
        self.data = np.matrix(vec)

    def define_norm_vectors(self):
        '''
        build matrix X and Y
        :return:
        '''
        X1 = self.data[:, :self.degf[0]]
        X2 = self.data[:, self.degf[0]:self.degf[1]]
        X3 = self.data[:, self.degf[1]:self.degf[2]]
        #matrix of vectors i.e.X = [[X11,X12],[X21],...]
        self.X = [X1, X2, X3]
        #number columns in matrix X
        self.mX = self.degf[2]
        # matrix, that consists of i.e. Y1,Y2
        self.Y = self.data[:, self.degf[2]:self.degf[3]]
        self.Y_ = self.datas[:, self.degf[2]:self.degf[3]]
        self.X_ = [self.datas[:, :self.degf[0]], self.datas[:,self.degf[0]:self.degf[1]],
                   self.datas[:, self.degf[1]:self.degf[2]]]

    def built_B(self):
        def B_average():
            '''
            Vector B as avarage of max and min in Y. B[i] =max Y[i,:]
            :return:
            '''
            b = np.tile((self.Y.max(axis=1) + self.Y.min(axis=1))/2,(1,self.deg[3]))
            return b

        def B_scaled():
            '''
            Vector B  = Y
            :return:
            '''
            return deepcopy(self.Y)

        if self.weights == 'average':
            self.B = B_average()
        elif self.weights =='scaled':
            self.B = B_scaled()
        else:
            exit('B not definded')

    def poly_func(self):
        '''
        Define function to polynoms
        :return: function
        '''
        if self.poly_type =='chebyshev':
            self.poly_f = special.eval_sh_chebyt
        elif self.poly_type == 'legendre':
            self.poly_f = special.eval_sh_legendre
        elif self.poly_type == 'laguerre':
            self.poly_f = special.eval_laguerre
        elif self.poly_type == 'hermit':
            self.poly_f = special.eval_hermite

    def built_A(self):
        '''
        built matrix A on shifted polynomys Chebysheva
        :param self.p:mas of deg for vector X1,X2,X3 i.e.
        :param self.X: it is matrix that has vectors X1 - X3 for example
        :return: matrix A as ndarray
        '''

        def mA():
            '''
            :param X: [X1, X2, X3]
            :param p: [p1,p2,p3]
            :return: m = m1*p1+m2*p2+...
            '''
            m = 0
            for i in range(len(self.X)):
                m+= self.X[i].shape[1]*(self.p[i]+1)
            return m

        def coordinate(v,deg):
            '''
            :param v: vector
            :param deg: chebyshev degree polynom
            :return:column with chebyshev value of coordiate vector
            '''
            c = np.ndarray(shape=(self.n,1), dtype = float)
            for i in range(self.n):
                c[i,0] = self.poly_f(deg, v[i])
            return c

        def vector(vec, p):
            '''
            :param vec: it is X that consist of X11, X12, ... vectors
            :param p: max degree for chebyshev polynom
            :return: part of matrix A for vector X1
            '''
            n, m = vec.shape
            a = np.ndarray(shape=(n,0),dtype = float)
            for j in range(m):
                for i in range(p):
                    ch = coordinate(vec[:,j],i)
                    a = np.append(a,ch,1)
            return a

        #k = mA()
        A = np.ndarray(shape = (self.n,0),dtype =float)
        for i in range(len(self.X)):
            vec = vector(self.X[i],self.p[i])
            A = np.append(A, vec,1)
        self.A = np.matrix(A)

    def lamb(self):
        lamb = np.ndarray(shape = (self.A.shape[1],0), dtype = float)
        for i in range(self.deg[3]):
            if self.splitted_lambdas:
                boundary_1 = self.p[0] * self.deg[0]
                boundary_2 = self.p[1] * self.deg[1] + boundary_1
                lamb1 = self._minimize_equation(self.A[:, :boundary_1], self.B[:, i])
                lamb2 = self._minimize_equation(self.A[:, boundary_1:boundary_2], self.B[:, i])
                lamb3 = self._minimize_equation(self.A[:, boundary_2:], self.B[:, i])
                lamb = np.append(lamb, np.concatenate((lamb1, lamb2, lamb3)), axis=1)
            else:
                lamb = np.append(lamb, self._minimize_equation(self.A, self.B[:, i]), axis=1)
        self.Lamb = np.matrix(lamb) #Lamb in full events

    def psi(self):
        def built_psi(lamb):
            '''
            return matrix xi1 for b1 as matrix
            :param A:
            :param lamb:
            :param p:
            :return: matrix psi, for each Y
            '''
            psi = np.ndarray(shape=(self.n, self.mX), dtype = float)
            q = 0 #iterator in lamb and A
            l = 0 #iterator in columns psi
            for k in range(len(self.X)): # choose X1 or X2 or X3
                for s in range(self.X[k].shape[1]):# choose X11 or X12 or X13
                    for i in range(self.X[k].shape[0]):
                            psi[i,l] = self.A[i,q:q+self.p[k]]*lamb[q:q+self.p[k], 0]
                    q+=self.p[k]
                    l+=1
            return np.matrix(psi)

        self.Psi = [] #as list because psi[i] is matrix(not vector)
        for i in range(self.deg[3]):
            self.Psi.append(built_psi(self.Lamb[:,i]))

    def built_a(self):
        self.a = np.ndarray(shape=(self.mX,0), dtype=float)
        for i in range(self.deg[3]):
            a1 = self._minimize_equation(self.Psi[i][:, :self.degf[0]], self.Y[:, i])
            a2 = self._minimize_equation(self.Psi[i][:, self.degf[0]:self.degf[1]], self.Y[:, i])
            a3 = self._minimize_equation(self.Psi[i][:, self.degf[1]:], self.Y[:, i])
            # temp = self._minimize_equation(self.Psi[i], self.Y[:, i])
            # self.a = np.append(self.a, temp, axis=1)
            self.a = np.append(self.a, np.vstack((a1, a2, a3)),axis = 1)

    def built_F1i(self, psi, a):
            '''
            not use; it used in next function
            :param psi: matrix psi (only one
            :param a: vector with shape = (6,1)
            :param degf:  = [3,4,6]//fibonachi of deg
            :return: matrix of (three) components with F1 F2 and F3
            '''
            m = len(self.X) # m  = 3
            F1i = np.ndarray(shape = (self.n,m),dtype = float)
            k = 0 #point of begining columnt to multipy
            for j in range(m): # 0 - 2
                for i in range(self.n): # 0 - 49
                    F1i[i,j] = psi[i,k:self.degf[j]]*a[k:self.degf[j],0]
                k = self.degf[j]
            return np.matrix(F1i)

    def built_Fi(self):
        self.Fi = []
        for i in range(self.deg[3]):
            self.Fi.append(self.built_F1i(self.Psi[i],self.a[:,i]))

    def built_c(self):
        self.c = np.ndarray(shape = (len(self.X),0),dtype = float)
        for i in range(self.deg[3]):
            self.c = np.append(self.c, conjugate_gradient_method(self.Fi[i].T*self.Fi[i], self.Fi[i].T*self.Y[:,i],self.eps),\
                          axis = 1)

    def built_F(self):
        F = np.ndarray(self.Y.shape, dtype = float)
        for j in range(F.shape[1]):#2
            for i in range(F.shape[0]): #50
                F[i,j] = self.Fi[j][i,:]*self.c[:,j]
        self.F = np.matrix(F)
        self.norm_error = []
        for i in range(self.Y.shape[1]):
            self.norm_error.append(np.linalg.norm(self.Y[:,i] - self.F[:,i],np.inf))

    def built_F_(self):
        minY = self.Y_.min(axis=0)
        maxY = self.Y_.max(axis=0)
        self.F_ = np.multiply(self.F,maxY - minY) + minY
        self.error = []
        for i in range(self.Y_.shape[1]):
            self.error.append(np.linalg.norm(self.Y_[:,i] - self.F_[:,i],np.inf))

    def save_to_file(self):
        wb = Workbook()
        #get active worksheet
        ws = wb.active

        l = [None]

        ws.append(['Вхідні дані: X'])
        for i in range(self.n):
             ws.append(l+self.datas[i,:self.degf[3]].tolist()[0])
        ws.append([])

        ws.append(['Вхідні дані: Y'])
        for i in range(self.n):
             ws.append(l+self.datas[i,self.degf[2]:self.degf[3]].tolist()[0])
        ws.append([])

        ws.append(['X нормалізовані:'])
        for i in range(self.n):
             ws.append(l+self.data[i,:self.degf[2]].tolist()[0])
        ws.append([])

        ws.append(['Y нормалізовані:'])
        for i in range(self.n):
             ws.append(l+self.data[i,self.degf[2]:self.degf[3]].tolist()[0])
        ws.append([])


        ws.append(['Матриця B:'])
        for i in range(self.n):
             ws.append(l+self.B[i].tolist()[0])
        ws.append([])

        ws.append(['Матриця A:'])
        for i in range(self.A.shape[0]):
             ws.append(l+self.A[i].tolist()[0])
        ws.append([])

        ws.append(['Матриця Lambda:'])
        for i in range(self.Lamb.shape[0]):
             ws.append(l+self.Lamb[i].tolist()[0])
        ws.append([])

        for j in range(len(self.Psi)):
             s = 'Матриця Psi%i:' %(j+1)
             ws.append([s])
             for i in range(self.n):
                  ws.append(l+self.Psi[j][i].tolist()[0])
             ws.append([])

        ws.append(['Матриця a:'])
        for i in range(self.mX):
             ws.append(l+self.a[i].tolist()[0])
        ws.append([])

        for j in range(len(self.Fi)):
             s = 'Матриця F%i:' %(j+1)
             ws.append([s])
             for i in range(self.Fi[j].shape[0]):
                  ws.append(l+self.Fi[j][i].tolist()[0])
             ws.append([])

        ws.append(['Матриця c:'])
        for i in range(len(self.X)):
             ws.append(l+self.c[i].tolist()[0])
        ws.append([])

        ws.append(['Y перебудовані нормалізовані :'])
        for i in range(self.n):
             ws.append(l+self.F[i].tolist()[0])
        ws.append([])

        ws.append(['Y перебудовані :'])
        for i in range(self.n):
             ws.append(l+self.F_[i].tolist()[0])
        ws.append([])

        ws.append(['Нормалізована похибка (Y - F)'])
        ws.append(l + self.norm_error)

        ws.append(['Похибка (Y_ - F_))'])
        ws.append(l+self.error)

        wb.save(self.filename_output)

    def show(self):
        text = []

        text.append('Input data: X')
        text.append(tb(np.array(self.datas[:, :self.degf[2]])))

        text.append('\nInput data: Y')
        text.append(tb(np.array(self.datas[:,self.degf[2]:self.degf[3]])))

        text.append('\nX normalised:')
        text.append(tb(np.array(self.data[:,:self.degf[2]])))

        text.append('\nY normalised:')
        text.append(tb(np.array(self.data[:,self.degf[2]:self.degf[3]])))

        text.append('\nmatrix B:')
        text.append(tb(np.array(self.B)))

        text.append('\nmatrix A:')
        text.append(tb(np.array(self.A)))

        text.append('\nmatrix Lambda:')
        text.append(tb(np.array(self.Lamb)))

        for j in range(len(self.Psi)):
             s = '\nmatrix Psi%i:' %(j+1)
             text.append(s)
             text.append(tb(np.array(self.Psi[j])))

        text.append('\nmatrix a:')
        text.append(tb(self.a.tolist()))

        for j in range(len(self.Fi)):
             s = '\nmatrix F%i:' %(j+1)
             text.append(s)
             text.append(tb(np.array(self.Fi[j])))

        text.append('\nmatrix c:')
        text.append(tb(np.array(self.c)))

        text.append('\nY rebuilt normalized :')
        text.append(tb(np.array(self.F)))

        text.append('\nY rebuilt :')
        text.append(tb(self.F_.tolist()))

        text.append('\nError normalised (Y - F)')
        text.append(tb([self.norm_error]))

        text.append('\nError (Y_ - F_))')
        text.append(tb([self.error]))

        return '\n'.join(text)

    def prepare(self):
        self.define_data()
        self.norm_data()
        self.define_norm_vectors()
        self.built_B()
        self.poly_func()
        self.built_A()
        self.lamb()
        self.psi()
        self.built_a()
        self.built_Fi()
        self.built_c()
        self.built_F()
        self.built_F_()
        self.save_to_file()
