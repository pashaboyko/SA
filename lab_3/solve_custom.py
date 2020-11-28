from copy import deepcopy

from tabulate import tabulate as tb
from math import pi

from lab_3.system_solve import *
from lab_3.solve import Solve


class SolveExpTh(Solve):

    def built_B(self):
        def B_average():
            """
            Vector B as average of max and min in Y. B[i] =max Y[i,:]
            :return:
            """
            b = np.tile((self.Y.max(axis=1) + self.Y.min(axis=1)) / 2, (1, self.dim[3]))
            return b

        def B_scaled():
            """
            Vector B  = Y
            :return:
            """
            return deepcopy(self.Y)

        if self.weights == 'average':
            self.B = B_average()
        elif self.weights == 'scaled':
            self.B = B_scaled()
        else:
            exit('B not defined')
        self.B_log = np.log(self.B + 1 + self.OFFSET)

    def built_A(self):
        """
        built matrix A on shifted polynomials Chebysheva
        :param self.p:mas of deg for vector X1,X2,X3 i.e.
        :param self.X: it is matrix that has vectors X1 - X3 for example
        :return: matrix A as ndarray
        """

        def mA():
            """
            :param X: [X1, X2, X3]
            :param p: [p1,p2,p3]
            :return: m = m1*p1+m2*p2+...
            """
            m = 0
            for i in range(len(self.X)):
                m += self.X[i].shape[1] * (self.deg[i] + 1)
            return m

        def coordinate(v, deg):
            """
            :param v: vector
            :param deg: chebyshev degree polynom
            :return:column with chebyshev value of coordiate vector
            """
            c = np.ndarray(shape=(self.n, 1), dtype=float)
            for i in range(self.n):
                c[i, 0] = self.poly_f(deg, v[i])
            return c

        def vector(vec, p):
            """
            :param vec: it is X that consist of X11, X12, ... vectors
            :param p: max degree for chebyshev polynom
            :return: part of matrix A for vector X1
            """
            n, m = vec.shape
            a = np.ndarray(shape=(n, 0), dtype=float)
            for j in range(m):
                for i in range(p):
                    ch = coordinate(vec[:, j], i)
                    a = np.append(a, ch, 1)
            return a

        # k = mA()
        A = np.ndarray(shape=(self.n, 0), dtype=float)
        for i in range(len(self.X)):
            vec = vector(self.X[i], self.deg[i])
            A = np.append(A, vec, 1)
        self.A_log = np.matrix(np.tanh(A))
        self.A = np.exp(self.A_log)

    def lamb(self):
        lamb = np.ndarray(shape=(self.A.shape[1], 0), dtype=float)
        for i in range(self.dim[3]):
            if self.splitted_lambdas:
                boundary_1 = self.deg[0] * self.dim[0]
                boundary_2 = self.deg[1] * self.dim[1] + boundary_1
                lamb1 = self._minimize_equation(self.A_log[:, :boundary_1], self.B_log[:, i])
                lamb2 = self._minimize_equation(self.A_log[:, boundary_1:boundary_2], self.B_log[:, i])
                lamb3 = self._minimize_equation(self.A_log[:, boundary_2:], self.B_log[:, i])
                lamb = np.append(lamb, np.concatenate((lamb1, lamb2, lamb3)), axis=1)
            else:
                lamb = np.append(lamb, self._minimize_equation(self.A_log, self.B_log[:, i]), axis=1)
        self.Lamb = np.matrix(lamb)  # Lamb in full events

    def psi(self):
        def built_psi(lamb):
            """
            return matrix xi1 for b1 as matrix
            :param A:
            :param lamb:
            :param p:
            :return: matrix psi, for each Y
            """
            psi = np.ndarray(shape=(self.n, self.mX), dtype=float)
            q = 0  # iterator in lamb and A
            l = 0  # iterator in columns psi
            for k in range(len(self.X)):  # choose X1 or X2 or X3
                for s in range(self.X[k].shape[1]):  # choose X11 or X12 or X13
                    for i in range(self.X[k].shape[0]):
                        psi[i, l] = self.A_log[i, q:q + self.deg[k]] * lamb[q:q + self.deg[k], 0]
                    q += self.deg[k]
                    l += 1
            return np.matrix(psi)

        self.Psi = list()
        self.Psi_tanh = list()
        for i in range(self.dim[3]):
            self.Psi.append(np.exp(built_psi(self.Lamb[:, i])) - 1)  # Psi = exp(sum(lambda*tanh(phi))) - 1
            self.Psi_tanh.append(np.tanh(self.Psi[-1]))


    def built_a(self):
        self.a = np.ndarray(shape=(self.mX, 0), dtype=float)
        for i in range(self.dim[3]):
            a1 = self._minimize_equation(self.Psi_tanh[i][:, :self.dim_integral[0]],
                                         np.log(self.Y[:, i] + 1 + self.OFFSET))
            a2 = self._minimize_equation(self.Psi_tanh[i][:, self.dim_integral[0]:self.dim_integral[1]],
                                         np.log(self.Y[:, i] + 1 + self.OFFSET))
            a3 = self._minimize_equation(self.Psi_tanh[i][:, self.dim_integral[1]:],
                                         np.log(self.Y[:, i] + 1 + self.OFFSET))
            # temp = self._minimize_equation(self.Psi[i], self.Y[:, i])
            # self.a = np.append(self.a, temp, axis=1)
            self.a = np.append(self.a, np.vstack((a1, a2, a3)), axis=1)

    def built_F1i(self, psi, a):
        """
        not use; it used in next function
        :param psi: matrix psi (only one
        :param a: vector with shape = (6,1)
        :param dim_integral:  = [3,4,6]//fibonacci of deg
        :return: matrix of (three) components with F1 F2 and F3
        """
        m = len(self.X)  # m  = 3
        F1i = np.ndarray(shape=(self.n, m), dtype=float)
        k = 0  # point of beginning column to multiply
        for j in range(m):  # 0 - 2
            for i in range(self.n):  # 0 - 49
                F1i[i, j] = psi[i, k:self.dim_integral[j]] * a[k:self.dim_integral[j], 0]
            k = self.dim_integral[j]
        return np.matrix(F1i)

    def built_Fi(self):
        self.Fi_tanh = list()
        self.Fi = list()
        for i in range(self.dim[3]):
            self.Fi.append(np.exp(self.built_F1i(self.Psi_tanh[i], self.a[:, i])) - 1)  # Fi = exp(sum(a*tanh(Psi))) - 1
            self.Fi_tanh.append(np.tanh(self.Fi[i]))

    def built_c(self):
        self.c = np.ndarray(shape=(len(self.X), 0), dtype=float)
        for i in range(self.dim[3]):
            self.c = np.append(self.c, self._minimize_equation(self.Fi_tanh[i], np.log(self.Y[:, i] + 1 + self.OFFSET))
                               , axis=1)

    def built_F(self):
        F = np.ndarray(self.Y.shape, dtype=float)
        for j in range(F.shape[1]):  # 2
            for i in range(F.shape[0]):  # 50
                F[i, j] = self.Fi_tanh[j][i, :] * self.c[:, j]
        self.F = np.exp(np.matrix(F)) - 1 - self.OFFSET  # F = exp(sum(c*tanh(Fi))) - 1
        self.norm_error = []
        for i in range(self.Y.shape[1]):
            self.norm_error.append(np.linalg.norm(self.Y[:, i] - self.F[:, i], np.inf))

    def aggregate(self, values, coeffs):
        return np.exp(np.dot(np.tanh(values), coeffs)) - 1

    def show(self):
        text = []
        text.append('\nError normalised (Y - F)')
        text.append(tb([self.norm_error]))

        text.append('\nError (Y_ - F_))')
        text.append(tb([self.error]))

        text.append('Input data: X')
        text.append(tb(np.array(self.datas[:, :self.dim_integral[2]])))

        text.append('\nInput data: Y')
        text.append(tb(np.array(self.datas[:, self.dim_integral[2]:self.dim_integral[3]])))

        text.append('\nX normalised:')
        text.append(tb(np.array(self.data[:, :self.dim_integral[2]])))

        text.append('\nY normalised:')
        text.append(tb(np.array(self.data[:, self.dim_integral[2]:self.dim_integral[3]])))

        text.append('\nmatrix B:')
        text.append(tb(np.array(self.B)))

        # text.append('\nmatrix A:')
        # text.append(tb(np.array(self.A)))

        text.append('\nmatrix Lambda:')
        text.append(tb(np.array(self.Lamb)))

        for j in range(len(self.Psi)):
            s = '\nmatrix Psi%i:' % (j + 1)
            text.append(s)
            text.append(tb(np.array(self.Psi[j])))

        text.append('\nmatrix a:')
        text.append(tb(self.a.tolist()))

        for j in range(len(self.Fi)):
            s = '\nmatrix F%i:' % (j + 1)
            text.append(s)
            text.append(tb(np.array(self.Fi[j])))

        text.append('\nmatrix c:')
        text.append(tb(np.array(self.c)))

        text.append('\nY rebuilt normalized :')
        text.append(tb(np.array(self.F)))

        text.append('\nY rebuilt :')
        text.append(tb(self.F_.tolist()))
        return '\n'.join(text)


class SolveExpTh1(Solve):

    def built_B(self):
        def B_average():
            """
            Vector B as average of max and min in Y. B[i] =max Y[i,:]
            :return:
            """
            b = np.tile((self.Y.max(axis=1) + self.Y.min(axis=1)) / 2, (1, self.dim[3]))
            return b

        def B_scaled():
            """
            Vector B  = Y
            :return:
            """
            return deepcopy(self.Y)

        if self.weights == 'average':
            self.B = B_average()
        elif self.weights == 'scaled':
            self.B = B_scaled()
        else:
            exit('B not defined')
        self.B_log = np.log(self.B + 1 + self.OFFSET)

    def built_A(self):
        """
        built matrix A on shifted polynomials Chebysheva
        :param self.p:mas of deg for vector X1,X2,X3 i.e.
        :param self.X: it is matrix that has vectors X1 - X3 for example
        :return: matrix A as ndarray
        """

        def mA():
            """
            :param X: [X1, X2, X3]
            :param p: [p1,p2,p3]
            :return: m = m1*p1+m2*p2+...
            """
            m = 0
            for i in range(len(self.X)):
                m += self.X[i].shape[1] * (self.deg[i] + 1)
            return m

        def coordinate(v, deg):
            """
            :param v: vector
            :param deg: chebyshev degree polynom
            :return:column with chebyshev value of coordiate vector
            """
            c = np.ndarray(shape=(self.n, 1), dtype=float)
            for i in range(self.n):
                c[i, 0] = self.poly_f(deg, v[i])
            return c

        def vector(vec, p):
            """
            :param vec: it is X that consist of X11, X12, ... vectors
            :param p: max degree for chebyshev polynom
            :return: part of matrix A for vector X1
            """
            n, m = vec.shape
            a = np.ndarray(shape=(n, 0), dtype=float)
            for j in range(m):
                for i in range(p):
                    ch = coordinate(vec[:, j], i)
                    a = np.append(a, ch, 1)
            return a

        # k = mA()
        A = np.ndarray(shape=(self.n, 0), dtype=float)
        for i in range(len(self.X)):
            vec = vector(self.X[i], self.deg[i])
            A = np.append(A, vec, 1)
        self.A_log = np.matrix(2/pi*np.arctan(A))
        self.A = np.exp(self.A_log)

    def lamb(self):
        lamb = np.ndarray(shape=(self.A.shape[1], 0), dtype=float)
        for i in range(self.dim[3]):
            if self.splitted_lambdas:
                boundary_1 = self.deg[0] * self.dim[0]
                boundary_2 = self.deg[1] * self.dim[1] + boundary_1
                lamb1 = self._minimize_equation(self.A_log[:, :boundary_1], self.B_log[:, i])
                lamb2 = self._minimize_equation(self.A_log[:, boundary_1:boundary_2], self.B_log[:, i])
                lamb3 = self._minimize_equation(self.A_log[:, boundary_2:], self.B_log[:, i])
                lamb = np.append(lamb, np.concatenate((lamb1, lamb2, lamb3)), axis=1)
            else:
                lamb = np.append(lamb, self._minimize_equation(self.A_log, self.B_log[:, i]), axis=1)
        self.Lamb = np.matrix(lamb)  # Lamb in full events

    def psi(self):
        def built_psi(lamb):
            """
            return matrix xi1 for b1 as matrix
            :param A:
            :param lamb:
            :param p:
            :return: matrix psi, for each Y
            """
            psi = np.ndarray(shape=(self.n, self.mX), dtype=float)
            q = 0  # iterator in lamb and A
            l = 0  # iterator in columns psi
            for k in range(len(self.X)):  # choose X1 or X2 or X3
                for s in range(self.X[k].shape[1]):  # choose X11 or X12 or X13
                    for i in range(self.X[k].shape[0]):
                        psi[i, l] = self.A_log[i, q:q + self.deg[k]] * lamb[q:q + self.deg[k], 0]
                    q += self.deg[k]
                    l += 1
            return np.matrix(psi)

        self.Psi = list()
        self.Psi_arctan = list()
        for i in range(self.dim[3]):
            self.Psi.append(np.exp(built_psi(self.Lamb[:, i])) - 1)  # Psi = exp(sum(lambda*2/pi*arctan(phi))) - 1
            self.Psi_arctan.append(2/pi*np.arctan(self.Psi[-1]))


    def built_a(self):
        self.a = np.ndarray(shape=(self.mX, 0), dtype=float)
        for i in range(self.dim[3]):
            a1 = self._minimize_equation(self.Psi_arctan[i][:, :self.dim_integral[0]],
                                         np.log(self.Y[:, i] + 1 + self.OFFSET))
            a2 = self._minimize_equation(self.Psi_arctan[i][:, self.dim_integral[0]:self.dim_integral[1]],
                                         np.log(self.Y[:, i] + 1 + self.OFFSET))
            a3 = self._minimize_equation(self.Psi_arctan[i][:, self.dim_integral[1]:],
                                         np.log(self.Y[:, i] + 1 + self.OFFSET))
            # temp = self._minimize_equation(self.Psi[i], self.Y[:, i])
            # self.a = np.append(self.a, temp, axis=1)
            self.a = np.append(self.a, np.vstack((a1, a2, a3)), axis=1)

    def built_F1i(self, psi, a):
        """
        not use; it used in next function
        :param psi: matrix psi (only one
        :param a: vector with shape = (6,1)
        :param dim_integral:  = [3,4,6]//fibonacci of deg
        :return: matrix of (three) components with F1 F2 and F3
        """
        m = len(self.X)  # m  = 3
        F1i = np.ndarray(shape=(self.n, m), dtype=float)
        k = 0  # point of beginning column to multiply
        for j in range(m):  # 0 - 2
            for i in range(self.n):  # 0 - 49
                F1i[i, j] = psi[i, k:self.dim_integral[j]] * a[k:self.dim_integral[j], 0]
            k = self.dim_integral[j]
        return np.matrix(F1i)

    def built_Fi(self):
        self.Fi_arctan = list()
        self.Fi = list()
        for i in range(self.dim[3]):
            self.Fi.append(np.exp(self.built_F1i(self.Psi_arctan[i], self.a[:, i])) - 1)  # Fi = exp(sum(a*2/pi*arctan(Psi))) - 1
            self.Fi_arctan.append(2/pi*np.arctan(self.Fi[i]))

    def built_c(self):
        self.c = np.ndarray(shape=(len(self.X), 0), dtype=float)
        for i in range(self.dim[3]):
            self.c = np.append(self.c, self._minimize_equation(self.Fi_arctan[i], np.log(self.Y[:, i] + 1 + self.OFFSET))
                               , axis=1)

    def built_F(self):
        F = np.ndarray(self.Y.shape, dtype=float)
        for j in range(F.shape[1]):  # 2
            for i in range(F.shape[0]):  # 50
                F[i, j] = self.Fi_arctan[j][i, :] * self.c[:, j]
        self.F = np.exp(np.matrix(F)) - 1 - self.OFFSET  # F = exp(sum(c*2/pi*arctan(Fi))) - 1
        self.norm_error = []
        for i in range(self.Y.shape[1]):
            self.norm_error.append(np.linalg.norm(self.Y[:, i] - self.F[:, i], np.inf))

    def show(self):
        text = []
        text.append('\nError normalised (Y - F)')
        text.append(tb([self.norm_error]))

        text.append('\nError (Y_ - F_))')
        text.append(tb([self.error]))

        # text.append('Input data: X')
        # text.append(tb(np.array(self.datas[:, :self.dim_integral[2]])))
        #
        # text.append('\nInput data: Y')
        # text.append(tb(np.array(self.datas[:, self.dim_integral[2]:self.dim_integral[3]])))
        #
        # text.append('\nX normalised:')
        # text.append(tb(np.array(self.data[:, :self.dim_integral[2]])))
        #
        # text.append('\nY normalised:')
        # text.append(tb(np.array(self.data[:, self.dim_integral[2]:self.dim_integral[3]])))
        #
        # text.append('\nmatrix B:')
        # text.append(tb(np.array(self.B)))
        #
        # # text.append('\nmatrix A:')
        # # text.append(tb(np.array(self.A)))
        #
        # text.append('\nmatrix Lambda:')
        # text.append(tb(np.array(self.Lamb)))
        #
        # for j in range(len(self.Psi)):
        #     s = '\nmatrix Psi%i:' % (j + 1)
        #     text.append(s)
        #     text.append(tb(np.array(self.Psi[j])))
        #
        # text.append('\nmatrix a:')
        # text.append(tb(self.a.tolist()))
        #
        # for j in range(len(self.Fi)):
        #     s = '\nmatrix F%i:' % (j + 1)
        #     text.append(s)
        #     text.append(tb(np.array(self.Fi[j])))
        #
        # text.append('\nmatrix c:')
        # text.append(tb(np.array(self.c)))
        #
        # text.append('\nY rebuilt normalized :')
        # text.append(tb(np.array(self.F)))
        #
        # text.append('\nY rebuilt :')
        # text.append(tb(self.F_.tolist()))
        return '\n'.join(text)