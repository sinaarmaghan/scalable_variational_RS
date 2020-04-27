# s_v statt q_U_sigma

import numpy as np
from data_parser import *
import matplotlib
import matplotlib.pyplot as plt
import cython
import FreeEnergy
import math
import sys
from termcolor import colored, cprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Cython.Build import cythonize
from distutils.core import setup
import time

class SVRS:

    def __init__(self, x, g, f, main_feat):
        self.Name = 'Start'

        if f.shape[1] != x.shape[0]:
            raise Exception('shape of f doesnt match x')

        self.X = x
        self.G = g
        self.F = f

        self.K = main_feat

        self.init_shapes()
        self.init_precisions()
        self.init_uvab()
        self.init_omega(0.0)
        self.init_q_sigma()
        self.init_r_matrix()
        self.details = []
        self.energy = []
        self.error = []
        self.prt_energy = True
        self.prt_error = True
        free_energy = self.calculate_entire_free_energy()
        print('\nFirst free energy ', free_energy)
        print('--------------------')

        self.details += [(self.Name, free_energy, ' percent change')]
        self.energy += [free_energy]

    def init_shapes(self):

        self.I, self.J = self.X.shape
        self.M = self.F.shape[0]
        self.N = self.G.shape[0]

    def init_precisions(self):

        self.x_tau = 100.0
        self.a_phi = 100.0
        self.u_alpha = 100.0
        self.v_beta = 100.0
        self.b_varphi = 100.0

        self.u_alpha_vector = self.u_alpha * np.ones(self.K)
        self.v_beta_vector = self.v_beta * np.ones(self.K)
        self.a_phi_vector = self.a_phi * np.ones(self.K)
        self.b_varphi_vector = self.b_varphi * np.ones(self.K)

    def init_uvab(self):

        # self.U = np.random.normal(0.0, 0.01, size=[self.K, self.I])
        # self.V = np.random.normal(0.0, 0.01, size=[self.K, self.J])

        e_u = np.random.normal(0.0, 0.01, size=[self.K, self.I])
        e_v = np.random.normal(0.0, 0.01, size=[self.K, self.J])

        self.A = np.random.normal(0.0, 1 / self.a_phi, size=[self.M, self.K])
        self.B = np.random.normal(0.0, 1 / self.b_varphi, size=[self.N, self.K])

        self.U = self.A.T.dot(self.F) + e_u
        self.V = self.B.T.dot(self.G) + e_v

    def init_q_sigma(self):
        self.S_U = np.full(shape=(self.K, self.I), fill_value=1 / self.u_alpha, dtype='float128')
        self.S_V = np.full(shape=(self.K, self.J), fill_value=1 / self.v_beta, dtype='float128')
        self.S_A = np.full(shape=(self.M, self.K), fill_value=1 / self.a_phi, dtype='float128')
        self.S_B = np.full(shape=(self.N, self.K), fill_value=1 / self.b_varphi, dtype='float128')

    def init_r_matrix(self):
        # self.R = self.X - np.matmul(np.transpose(self.U), self.V)
        self.R = self.X - self.U.T.dot(self.V)

        self.R_u = self.U - self.A.T.dot(self.F)

        self.R_v = self.V - self.B.T.dot(self.G)

    def init_omega(self, noneVal):
        m, n = self.X.shape
        omega_j = {}
        omega_i = {}
        omega = []
        for i in range(m):
            omega_i_list = []
            for j in range(n):
                if self.X[i, j] != noneVal:
                    omega += [(i, j)]
                    omega_i_list += [j]

            omega_i[i] = omega_i_list

        for j in range(n):
            omega_j_list = []
            for i in range(m):
                if self.X[i, j] != noneVal:
                    omega_j_list.append(i)

            omega_j.update({j: omega_j_list})

        self.omega = omega
        self.omega_i = omega_i
        self.omega_j = omega_j

    '''
    def calculate_free_energy(self):

        x_energy = 0.0
        u_energy = 0.0
        v_energy = 0.0
        a_energy = 0.0
        b_energy = 0.0

        # Todo None should probably be changed by -1 or something similar
        for i in range(self.I):
            for j in range(self.J):
                if self.X[i, j] is not None:
                    x_energy += self.calc_free_energy_x(i, j)

        for k in range(self.K):
            for i in range(self.I):
                u_energy += self.calc_free_energy_u(k, i)

        for k in range(self.K):
            for j in range(self.J):
                v_energy += self.calc_free_energy_v(k, j)

        for m in range(self.M):
            for k in range(self.K):
                a_energy += self.calc_free_energy_a(m, k)

        for n in range(self.N):
            for k in range(self.K):
                b_energy += self.calc_free_energy_b(n, k)

        # print_energy(x_energy, u_energy, v_energy, a_energy, b_energy)

        return np.abs(x_energy), np.abs(u_energy)[0], np.abs(v_energy)[0], np.abs(a_energy)[0], np.abs(b_energy)[0]
    '''

    def calculate_entire_free_energy(self):
        a_energy = self.calc_free_energy_mat_a()
        u_energy = self.calc_free_energy_mat_u()
        v_energy = self.calc_free_energy_mat_v()
        b_energy = self.calc_free_energy_mat_b()
        x_energy = self.calc_free_energy_mat_x()

        sum = a_energy + u_energy + v_energy + b_energy + x_energy
        sum *= -1
        '''

        print("Free energy of X = ", x_energy)
        print("Free energy of U = ", u_energy)
        print("Free energy of V = ", v_energy)
        print("Free energy of A = ", a_energy)
        print("Free energy of B = ", b_energy)
        print("---------------------------\n")

        '''
        return sum

    def calc_free_energy_mat_u(self) -> np.float128:
        cdef float gesamt_u_energy = 0.0
        cdef int k
        cdef int i
        for k in range(self.K):
            for i in range(self.I):
                gesamt_u_energy += self.calc_free_energy_u(k, i)

        return gesamt_u_energy

    def calc_free_energy_mat_v(self) -> np.float128:
        cdef float gesamt_v_energy = 0.0
        cdef int k
        cdef int j

        for k in range(self.K):
            for j in range(self.J):
                gesamt_v_energy += self.calc_free_energy_v(k, j)

        return gesamt_v_energy

    def calc_free_energy_mat_a(self) -> np.float128:
        cdef float gesamt_a_energy = 0.0
        cdef int m
        cdef int k
        for m in range(self.M):
            for k in range(self.K):
                gesamt_a_energy += self.calc_free_energy_a(m, k)

        return gesamt_a_energy

    def calc_free_energy_mat_b(self) -> np.float128:
        cdef float gesamt_b_energy = 0.0
        cdef int n
        cdef int k
        for n in range(self.N):
            for k in range(self.K):
                gesamt_b_energy += self.calc_free_energy_b(n, k)
        return gesamt_b_energy

    def calc_free_energy_mat_x(self) -> np.float128:
        cdef float gesamt_x_energy = 0.0

        # for i in range(self.I):
        #     for j in range(self.J):
        #         if self.X[i, j] is not None:
        #             gesamt_x_energy += self.calc_free_energy_x(i, j)

        for t in self.omega:
            gesamt_x_energy += self.calc_free_energy_x(*t)
        return gesamt_x_energy

    def calc_free_energy_x(self, i, j) -> np.float128:

        # Todo: if self.x_tau(division by zero)

        return -0.5 * self.x_tau * (self.R[i, j] ** 2 + self.w_ij(i, j)) + \
               0.5 * np.log(self.x_tau)

    '''
    def r_ij(self, i, j):

        return self.X[i, j] - np.dot(self.U[:, i], self.V[:, j])

    '''

    def w_ij(self, i, j):

        return np.dot(self.U[:, i] ** 2, self.S_V[:, j]) \
               + np.dot(self.V[:, j] ** 2, self.S_U[:, i]) \
               + np.dot(self.S_U[:, i], self.S_V[:, j])

    def calc_free_energy_u(self, k, i):
        val = -self.u_alpha_vector[k] / 2 * \
              ((self.U[k, i] - np.dot(self.A[:, k], self.F[:, i])) ** 2
               + self.S_U[k, i] +
               np.dot(self.S_A[:, k], self.F[:, i] ** 2)) \
              + 0.5 * np.log(self.S_U[k, i] * self.u_alpha_vector[k]) + 0.5

        return val

    def calc_free_energy_v(self, k, j):

        return -self.v_beta_vector[k] / 2 * \
               ((self.V[k, j] - np.dot(self.B[:, k], self.G[:, j])) ** 2
                + self.S_V[k, j] +
                np.dot(self.S_B[:, k], self.G[:, j] ** 2)) \
               + 0.5 * np.log(self.S_V[k, j] * self.v_beta_vector[k]) + 0.5

    def calc_free_energy_a(self, m, k):

        return -self.a_phi_vector[k] / 2 * (self.A[m, k] ** 2 + self.S_A[m, k]) + \
               0.5 * np.log(self.S_A[m, k] * self.a_phi_vector[k]) + 0.5

    def calc_free_energy_b(self, n, k):

        return -self.b_varphi_vector[k] / 2 * (self.B[n, k] ** 2 + self.S_B[n, k]) + \
               0.5 * np.log(self.S_B[n, k] * self.b_varphi_vector[k]) + 0.5

    def update_U(self):
        t0= time.clock()


        cdef int i
        cdef int k
        self.Name = " U "
        cdef int j
        cdef float summe_1 =0.0
        cdef float summe_2= 0.0
        'update_q_u'

        for i in range(self.I):

            for k in range(self.K):

                summe_1 = 0.0
                summe_2 = 0.0

                ones_array= np.ones((len(self.omega_i[i]),),dtype=int)

                kj=(ones_array*k, self.omega_i[i])
                ij = (ones_array*i, self.omega_i[i])
                ki = (ones_array*k,ones_array*i )

                ksi = self.U[k, i]

                #for j in self.omega_i[i]:
                #    summe_1 += self.V[k, j] ** 2 + self.S_V[k, j]


                summe_1 = np.sum(self.V[kj] ** 2 + self.S_V[kj])

                #if not math.isclose(summe3,summe_1):
                #    print(summe3,summe_1)
                #    input("")
                #t6 = time.clock()-t5
                #buff2+=t6
                self.S_U[k, i] = 1 / (self.u_alpha_vector[k] + self.x_tau * summe_1)


               # for j in self.omega_i[i]:
               #     summe_2 += (self.R[i, j] + self.U[k, i] * self.V[k, j]) * self.V[k, j]

                summe_2 = np.sum((self.R[ij] + self.U[ki]*self.V[kj])*self.V[kj])

                #if not math.isclose(summe4, summe_2):
                #    print(summe4,summe_2)
                #    input("")

                theta = self.u_alpha_vector[k] * np.dot(self.A[:, k], self.F[:, i]) + self.x_tau * summe_2

                self.U[k, i] = self.S_U[k, i] * theta



               # for j in self.omega_i[i]:
               #     self.R[i, j] = self.R[i, j] - (self.U[k, i] - ksi) * self.V[k, j]


                self.R[ij] = self.R[ij] - (self.U[ki]- ksi) * self.V[kj]


        t1 = time.clock()-t0



       # print(self.Name,"\t time needed:\t",t1)
       # input("Stop! ")
        self.check_energy(self.prt_energy)

    def update_V(self):
        cdef int j
        cdef int k
        self.Name = " V "
        cdef int i

        t0 = time.clock()
        'update q_v'
        for j in range(self.J):

            for k in range(self.K):

                summe_1 = 0.0

                summe_2 = 0.0

                ksi = self.V[k, j]


                for i in self.omega_j[j]:
                    summe_1 += self.U[k, i] ** 2 + self.S_U[k, i]



                self.S_V[k, j] = 1 / (self.v_beta_vector[k] + self.x_tau * summe_1)

                for i in self.omega_j[j]:
                    summe_2 += (self.R[i, j] + self.U[k, i] * self.V[k, j]) * self.U[k, i]

                theta = self.v_beta_vector[k] * np.dot(self.B[:, k], self.G[:, j]) + self.x_tau * summe_2

                self.V[k, j] = self.S_V[k, j] * theta

                for i in self.omega_j[j]:
                    self.R[i, j] = self.R[i, j] - self.U[k, i] * (self.V[k, j] - ksi)
        t1 = time.clock() - t0
       # print(self.Name,"\t time needed:\t",t1)
        self.check_energy(self.prt_energy)

    def update_A(self):
        cdef int k
        cdef int m
        self.Name = " A "
        t0 =time.clock()
        for k in range(self.K):
            for m in range(self.M):
                ksi = self.A[m, k]

                Samk = 1 / (
                        self.a_phi_vector[k] + self.u_alpha_vector[k] * np.linalg.norm(self.F[m, :]) ** 2)

                self.A[m, k] = Samk * self.u_alpha_vector[k] * \
                               (self.R_u[k, :] + self.A[m, k] * self.F[m, :]).dot(self.F[m, :].T)

                self.S_A[m, k] = Samk
                self.R_u[k, :] = self.R_u[k, :] - (self.A[m, k] - ksi) * self.F[m, :]
        t1 = time.clock() - t0
       # print(self.Name,"\t time needed:\t",t1)
        self.check_energy(self.prt_energy)

    def update_B(self):
        cdef int k
        cdef int n
        self.Name = " B "
        t0= time.clock()
        for k in range(self.K):
            for n in range(self.N):
                ksi = self.B[n, k]

                Sbnk = 1 / (
                        self.b_varphi_vector[k] + self.v_beta_vector[k] * np.linalg.norm(self.G[n, :]) ** 2)

                self.B[n, k] = Sbnk * self.v_beta_vector[k] * \
                               (self.R_v[k, :] + self.B[n, k] * self.G[n, :]).dot(self.G[n, :].T)

                self.S_B[n, k] = Sbnk
                self.R_v[k, :] = self.R_v[k, :] - (self.B[n, k] - ksi) * self.G[n, :]
        t1 = time.clock() - t0
       # print(self.Name,"\t time needed:\t",t1)
        self.check_energy(self.prt_energy)

    def update_hyper(self):
        cdef int k
        cdef int i
        cdef int j
        cdef int m
        cdef int n
        cdef float up_tau = 0.0
        cdef float up_alpha = 0.0
        cdef float up_beta = 0.0
        cdef float up_phi = 0.0

        self.Name = ' Hyper '

        'update tau'

        for tupel in self.omega:
            up_tau += self.R[tupel[0], tupel[1]] ** 2 + self.w_ij(tupel[0], tupel[1])

        self.x_tau = len(self.omega) / up_tau

        for k in range(self.K):

            'update alpha'
            up_alpha = 0.0

            for i in range(self.I):
                up_alpha += (self.U[k, i] - np.dot(self.A[:, k], self.F[:, i])) ** 2 \
                            + self.S_U[k, i] + \
                            np.dot(self.S_A[:, k], self.F[:, i] ** 2)

            self.u_alpha_vector[k] = self.I / up_alpha

            'update beta'
            up_beta = 0.0

            for j in range(self.J):
                up_beta += (self.V[k, j] - np.dot(self.B[:, k], self.G[:, j])) ** 2 \
                           + self.S_V[k, j] + \
                           np.dot(self.S_B[:, k], self.G[:, j] ** 2)

            self.v_beta_vector[k] = self.J / up_beta

            'update phi'
            up_phi = 0.0

            for m in range(self.M):
                up_phi += self.A[m, k] ** 2 + self.S_A[m, k]

            self.a_phi_vector[k] = self.M / up_phi

            'update varphi'
            varphi_up = 0.0

            for n in range(self.N):
                varphi_up += self.B[n, k] ** 2 + self.S_B[n, k]

            self.b_varphi_vector[k] = self.N / varphi_up

        self.check_energy(self.prt_energy)

    def check_energy(self, prt):
        self.Name="check Energy"
        t0= time.clock()


        before_up_energy = self.energy[-1]
        after_up_energy = self.calculate_entire_free_energy()
        convergence = math.isclose(after_up_energy, before_up_energy)

        diff = after_up_energy - before_up_energy

        percent_diff = "%.2f" % (-100.0 * diff / before_up_energy)

        self.details += [(self.Name, after_up_energy, 'Difference:   ' + str(diff),
                          '   Changes achieved:   ' + percent_diff + '%')]
        self.energy += [after_up_energy]

        if prt:
            if diff > 0 and not convergence:
                print('\nError in Update %s:\nBefore iteration = \t%f \nAfter iteration =\t %f'
                      % (self.Name, before_up_energy, after_up_energy))
                print('The difference is = %f\n' % (diff))

            if diff > 0:

                print('\x1b[6;30;42m' + "Free energy after updating:" + self.Name, "\t", after_up_energy, "\tDiff:\t",
                      after_up_energy - before_up_energy, "\tChanges:", percent_diff, "%" + '\x1b[0m')

            else:
                print("Free energy after updating:" + self.Name, "\t", after_up_energy, "\tDiff:\t",
                      after_up_energy - before_up_energy, "\tChanges:", percent_diff, "%")

        t1 = time.clock() - t0

        #print(self.Name,"\t time needed:\t",t1)


    def train(self, testdata, idx, iteration_num):

        for t in range(iteration_num):

            print(t, 'iteration')
            t0= time.clock()

            self.update_u_matrix()
            self.update_v_matrix()
            self.update_A()
            self.update_B()
            self.update_hyper_parameters()
            t1 = time.clock() - t0
            print("\t Iteration time:\t",t1)
            if self.prt_error:
                print("\nRMSE:\t", self.calc_error(testdata, idx), "x_tau:\t", self.x_tau)

            print("\n....................")

    def calc_error(self, test, test_idx):

        prediction = self.U.T.dot(self.V)
        error = math.sqrt(mean_squared_error(test[test_idx], prediction[test_idx]))

        self.error += [error]

        return error


def print_details(list):
    buff = 0

    print(0, ". Iteration \n")

    for element in list:
        print(element)
        if element[0] == ' Hyper ':
            print("................\n")
            buff = buff + 1
            if buff != len(list):
                print(buff, ". Iteration \n")

    print("--------------------\n\n")


def generate_normal_data(mitarbeiter, feat_mitarbeiter, projekte, feat_projekte, feat_main, sparsity):
    i = mitarbeiter
    j = projekte
    m = feat_mitarbeiter
    n = feat_projekte
    k = feat_main

    amat = np.random.normal(0.0, 0.1, size=[m, k])
    fmat = np.random.normal(0.0, 1.0, size=[m, i])
    bmat = np.random.normal(0.0, 1.0, size=[n, k])
    gmat = np.random.normal(0.0, 0.1, size=[n, j])

    e_u = np.random.normal(0.0, 0.1, size=[k, i])
    e_v = np.random.normal(0.0, 0.1, size=[k, j])

    u = amat.T.dot(fmat) + e_u
    v = bmat.T.dot(gmat) + e_v

    e_x = np.random.normal(0.0, 5, size=[i, j])

    x = u.T.dot(v) + e_x

    x = sparse_maker(x, percentage=sparsity)

    return amat, fmat, u, bmat, gmat, v, x



def sparse_maker(mat, percentage):
    mat_copy = mat.copy()
    set_to_zero = nonzeros(mat_copy, shuffle=True, percentage=percentage)
    mat_copy[set_to_zero] = 0.0
    return mat_copy


def train_test(mat, percentage):
    test_data = sparse_maker(mat, percentage=percentage)
    train_data = mat - test_data
    return train_data, test_data


def nonzeros(matrix, shuffle, percentage):
    if not shuffle:
        return np.nonzero(matrix)

    elif shuffle:
        nnz = np.nonzero(matrix)
        toshuffle = list(zip(nnz[0], nnz[1]))
        np.random.shuffle(toshuffle)
        if percentage is not None:
            scope = int(0.01 * percentage * len(toshuffle))
            if scope == 0:
                raise Exception("Bad Percentage")
            toshuffle = toshuffle[:scope]
        a, b = zip(*toshuffle)
        ret_obj = (np.array(a), np.array(b))

        return ret_obj

    else:
        raise Exception("please set arguments")



def print_predictions(test_idx):
    return None



def split_analyze_raw_data(x, percentage):
    nonzx = nonzeros(x, shuffle=False, percentage=None)
    print("variance of Data (Nonzeros)\t", np.var(x[nonzx]), "\nMaximum:\t", np.max(x), "\nMinimum:\t",
          np.min(x[nonzx]))
    print("................................................................")
    train, test = train_test(x, percentage=80)

    test_id = nonzeros(test, shuffle=False, percentage=None)

    print("Variance of Test data:\t", np.var(test[test_id]))
    print("Number of test Ratings:\t", len(test_id[0]))
    train_id = nonzeros(train, shuffle=False, percentage=None)
    print(".................................................................")
    print("Variance of Train data:", np.var(train[train_id]))
    print("Number of Train Ratings:", len(train_id[0]))
    print(".................................................................")
    input("")

    return train, test, test_id




