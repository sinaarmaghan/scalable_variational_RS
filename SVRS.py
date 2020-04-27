
from data_parser import *
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import plot


class SVRS:

    def __init__(self, train_data, test_data, g, f, main_features, columnwise, dim_reduction):
        self.Name = 'Start'
        self.columnwise = columnwise
        if f.shape[1] != train_data.shape[0]:
            raise Exception('shape of f doesnt match x')

        self.X = train_data
        self.test_data = test_data
        self.G = g
        self.F = f

        self.K = main_features
        self.extra_dim = set()
        self.dim_reduction = dim_reduction

        self.init_shapes()
        self.init_precisions()
        self.init_uvab()
        self.init_omega(0.0)
        self.init_q_sigma()
        self.init_r_matrix()
        self.details = []
        self.energy = []
        self.train_error = []
        self.test_error = []
        self.u_error = []
        self.print_energy = True
        self.print_error = True
        free_energy = self.calculate_entire_free_energy()
        print('\nFirst free energy ', free_energy)
        print('..................................')

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

        self.A = np.random.normal(0.2, 0.75 + 0 * self.a_phi, size=[self.M, self.K])
        self.B = np.random.normal(0.2, 0.75 + 0 * self.b_varphi, size=[self.N, self.K])
        self.U = self.A.T.dot(self.F)
        self.V = self.B.T.dot(self.G)

    def init_q_sigma(self):
        self.S_U = np.full(shape=(self.K, self.I), fill_value=1 / self.u_alpha, dtype='float128')
        self.S_V = np.full(shape=(self.K, self.J), fill_value=1 / self.v_beta, dtype='float128')
        self.S_A = np.full(shape=(self.M, self.K), fill_value=1 / self.a_phi, dtype='float128')
        self.S_B = np.full(shape=(self.N, self.K), fill_value=1 / self.b_varphi, dtype='float128')

    def init_r_matrix(self):
        self.R = self.X - self.U.T.dot(self.V)
        self.R_u = self.U - self.A.T.dot(self.F)
        self.R_v = self.V - self.B.T.dot(self.G)

    def init_omega(self, non_value):
        m, n = self.X.shape
        omega_j = {}
        omega_i = {}
        omega = []
        for i in range(m):
            omega_i_list = []
            for j in range(n):
                if self.X[i, j] != non_value:
                    omega += [(i, j)]
                    omega_i_list += [j]

            omega_i[i] = omega_i_list

        for j in range(n):
            omega_j_list = []
            for i in range(m):
                if self.X[i, j] != non_value:
                    omega_j_list.append(i)

            omega_j.update({j: omega_j_list})

        self.omega = omega
        self.omega_i = omega_i
        self.omega_j = omega_j

    def calculate_entire_free_energy(self):
        a_energy = self.calc_free_energy_mat_a()
        u_energy = self.calc_free_energy_mat_u()
        v_energy = self.calc_free_energy_mat_v()
        b_energy = self.calc_free_energy_mat_b()
        x_energy = self.calc_free_energy_mat_x()

        sum = a_energy + u_energy + v_energy + b_energy + x_energy
        sum *= -1

        return sum

    def calc_free_energy_mat_u(self) -> np.float128:
        gesamt_u_energy = 0.0
        for k in range(self.K):
            for i in range(self.I):
                gesamt_u_energy += self.calc_free_energy_u(k, i)

        return gesamt_u_energy

    def calc_free_energy_mat_v(self) -> np.float128:
        gesamt_v_energy = 0.0

        for k in range(self.K):
            for j in range(self.J):
                gesamt_v_energy += self.calc_free_energy_v(k, j)

        return gesamt_v_energy

    def calc_free_energy_mat_a(self) -> np.float128:
        gesamt_a_energy = 0.0
        for m in range(self.M):
            for k in range(self.K):
                gesamt_a_energy += self.calc_free_energy_a(m, k)

        return gesamt_a_energy

    def calc_free_energy_mat_b(self) -> np.float128:
        gesamt_b_energy = 0.0

        for n in range(self.N):
            for k in range(self.K):
                gesamt_b_energy += self.calc_free_energy_b(n, k)
        return gesamt_b_energy

    def calc_free_energy_mat_x(self) -> np.float128:
        gesamt_x_energy = 0.0

        # for i in range(self.I):
        #     for j in range(self.J):
        #         if self.X[i, j] is not None:
        #             gesamt_x_energy += self.calc_free_energy_x(i, j)

        for t in self.omega:
            gesamt_x_energy += self.calc_free_energy_x(*t)
        return gesamt_x_energy

    def calc_free_energy_x(self, i, j) -> np.float128:
        return -0.5 * self.x_tau * (self.R[i, j] ** 2 + self.w_ij(i, j)) + \
               0.5 * np.log(self.x_tau)

    def w_ij(self, i, j):

        return np.dot(self.U[:, i] ** 2, self.S_V[:, j]) \
               + np.dot(self.V[:, j] ** 2, self.S_U[:, i]) \
               + np.dot(self.S_U[:, i], self.S_V[:, j])

    def calc_free_energy_u(self, k, i):
        val = -0.5 * self.u_alpha_vector[k] * \
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

        return -0.5 * self.a_phi_vector[k] * (self.A[m, k] ** 2 + self.S_A[m, k]) + \
               0.5 * np.log(self.S_A[m, k] * self.a_phi_vector[k]) + 0.5

    def calc_free_energy_b(self, n, k):

        return -self.b_varphi_vector[k] / 2 * (self.B[n, k] ** 2 + self.S_B[n, k]) + \
               0.5 * np.log(self.S_B[n, k] * self.b_varphi_vector[k]) + 0.5

    def update_u_matrix(self):

        self.Name = " U "

        'update_q_u'

        for i in range(self.I):

            for k in range(self.K):

                summe_1 = 0.0

                summe_2 = 0.0

                ksi = np.copy(self.U[k, i])

                for j in self.omega_i[i]:
                    summe_1 += self.V[k, j] ** 2 + self.S_V[k, j]

                self.S_U[k, i] = 1 / (self.u_alpha_vector[k] + self.x_tau * summe_1)

                for j in self.omega_i[i]:
                    summe_2 += (self.R[i, j] + self.U[k, i] * self.V[k, j]) * self.V[k, j]

                theta = self.u_alpha_vector[k] * np.dot(self.A[:, k], self.F[:, i]) + self.x_tau * summe_2

                self.U[k, i] = self.S_U[k, i] * theta

                for j in self.omega_i[i]:
                    self.R[i, j] = self.R[i, j] - (self.U[k, i] - ksi) * self.V[k, j]

        self.check_energy(self.print_energy)

    def update_v_matrix(self):

        self.Name = " V "

        'update q_v'
        for j in range(self.J):

            for k in range(self.K):

                summe_1 = 0.0

                summe_2 = 0.0

                ksi = np.copy(self.V[k, j])

                for i in self.omega_j[j]:
                    summe_1 += self.U[k, i] ** 2 + self.S_U[k, i]

                self.S_V[k, j] = 1 / (self.v_beta_vector[k] + self.x_tau * summe_1)

                for i in self.omega_j[j]:
                    summe_2 += (self.R[i, j] + self.U[k, i] * self.V[k, j]) * self.U[k, i]

                theta = self.v_beta_vector[k] * np.dot(self.B[:, k], self.G[:, j]) + self.x_tau * summe_2

                self.V[k, j] = self.S_V[k, j] * theta

                for i in self.omega_j[j]:
                    self.R[i, j] = self.R[i, j] - self.U[k, i] * (self.V[k, j] - ksi)

        self.check_energy(self.print_energy)

    def update_a_matrix(self):

        self.Name = " A "

        for k in range(self.K):
            for m in range(self.M):
                self.S_A[m, k] = 1 / (
                        self.a_phi_vector[k] +
                        self.u_alpha_vector[k] * sum(self.F[m, :] ** 2))

                self.A[m, k] = self.S_A[m, k] * self.u_alpha_vector[k] * \
                               (self.U[k, :] - (self.A[:, k].T.dot(self.F) - self.A[m, k] * self.F[m, :])).dot(
                                   self.F[m, :].T)

        self.check_energy(self.print_energy)

    def update_b_matrix(self):

        self.Name = " B "

        for k in range(self.K):
            for n in range(self.N):
                self.S_B[n, k] = 1 / (
                        self.b_varphi_vector[k] + self.v_beta_vector[k] * sum(self.G[n, :] ** 2))

                self.B[n, k] = self.S_B[n, k] * self.v_beta_vector[k] * \
                               (self.V[k, :] - (self.B[:, k].T.dot(self.G) - self.B[n, k] * self.G[n, :])).dot(
                                   self.G[n, :].T)

        self.check_energy(self.print_energy)

    def update_hyper_parameters(self):

        self.Name = ' Hyper '

        'update tau'
        up_tau = 0.0

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

        self.check_energy(self.print_energy)

    def check_energy(self, prt):
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

    def train(self, iteration_num, g_test):

        for t in range(iteration_num):
            print("\n....................................")
            print("Iteration: ", t)
            print("\n....................................")

            if t > 50 and self.dim_reduction:
                self.dim_reduce(0.08)
                print(t)

            self.update_u_matrix()
            self.update_v_matrix()

            self.update_a_matrix()
            self.update_b_matrix()

            self.update_hyper_parameters()

            if self.print_error:
                train_error = self.calc_error(self.X, g_test, testdata=False)
                test_error = self.calc_error(self.test_data, g_test, testdata=True)
                print("RMSE:  \nTrain: ", train_error, "Test :", test_error)
                self.train_error += [train_error]
                self.test_error += [test_error]

        self.error_print()

    def error_print(self):
        buff = 0.0
        print("Trainerror:\n")
        for error in self.train_error:
            print(error, "\t Diff:", error - buff)
            buff = error

        buff = 0.0

        print("\nTesterror:\n")
        for error in self.test_error:
            print(error, "\t Diff:", error - buff)
            buff = error

        buff = (0.0, 0.0)

        # print("\nuerror:\n")
        # for error in self.uerror:
        #     print(error, "\t Diff:", error[0] - buff[0])
        #     buff = error

    def calc_error(self, data, g_test, testdata):

        nonz = np.nonzero(data)

        if g_test is not None and testdata:

            v_matrix = self.B.T.dot(g_test)
            prediction = self.U.T.dot(v_matrix)
            return math.sqrt(mean_squared_error(data[nonz], prediction[nonz]))

        else:

            prediction = self.U.T.dot(self.V)

            error = math.sqrt(mean_squared_error(data[nonz], prediction[nonz]))

            # error = np.sqrt(np.sum(self.R**2)/np.prod(self.R.shape))

            return error

    def calc_error_u(self, test):

        predictionu = self.A.T.dot(self.F)

        error = math.sqrt(mean_squared_error(self.U, test))

        error2 = math.sqrt(mean_squared_error(predictionu, test))

        return error, error2

    def calc_error_two_mat(self, test, prediction):

        error = math.sqrt(mean_squared_error(test, prediction))

        return error

    def print_predictions(self):

        rmat = self.U.T.dot(self.V)

        idx = nonzeros(self.test_data, shuffle=True, percentage=5)

        print(rmat[idx])
        print("\n")
        print(self.test_data[idx])

    def dim_reduce(self, eps):
        k = 0
        print("Dim Reduce-- ", k, self.K)
        print(1 / np.max(self.u_alpha_vector))
        print(1 / np.max(self.a_phi_vector))
        print(1 / np.max(self.v_beta_vector))
        print(1 / np.max(self.b_varphi_vector))
        print("----------------------------")

        while k < self.K:
            print(k, self.K)

            # print("alpha ", 1 / self.u_alpha_vector[k], "phi", 1 / self.a_phi_vector[k])
            if (1 / self.u_alpha_vector[k] < eps and 1 / self.a_phi_vector[k] < eps):
                print("------A----------------------")
                print(self.A[:, k])
                print("----U------------------------")
                print(self.U[k, :])
                print(self.F)
                print("\n---ATF------\n")
                print(self.A[:, k].T.dot(self.F))
                print("-------DIFF-------------------")
                print(self.U[k, :] - self.A[:, k].T.dot(self.F))
                print("--------------END OF U---------------")

            if (1 / self.v_beta_vector[k] < eps and 1 / self.b_varphi_vector[k] < eps):
                print("-----B-----------------------")
                print(self.B[:, k])
                print("----V------------------------")
                print(self.V[k, :])
                print("----------G------------------")
                print("\n---BTG------\n")
                print(self.B[:, k].T.dot(self.G))
                print("------Diff----------------------")
                print(self.V[k, :] - self.B[:, k].T.dot(self.G))
                print("--------------END OF V---------------")

            if (1 / self.u_alpha_vector[k] < eps and 1 / self.a_phi_vector[k] < eps) \
                    or (1 / self.v_beta_vector[k] < eps and 1 / self.b_varphi_vector[k] < eps):

                if not self.extra_dim.__contains__(k):
                    self.K = self.K - 1
                    self.extra_dim.add(k)

                    self.A = np.delete(self.A, k, 1)
                    self.S_A = np.delete(self.S_A, k, 1)
                    self.U = np.delete(self.U, k, 0)
                    self.S_U = np.delete(self.S_U, k, 0)
                    self.B = np.delete(self.B, k, 1)
                    self.S_B = np.delete(self.S_B, k, 1)
                    self.V = np.delete(self.V, k, 0)
                    self.S_V = np.delete(self.S_V, k, 0)

                    self.u_alpha_vector = np.delete(self.u_alpha_vector, k)
                    self.a_phi_vector = np.delete(self.a_phi_vector, k)
                    self.v_beta_vector = np.delete(self.v_beta_vector, k)
                    self.b_varphi_vector = np.delete(self.b_varphi_vector, k)

                    print(self.U.shape)
                    print(self.V.shape)
                    print(self.S_U.shape)
                    print(self.S_V.shape)
                    print(self.A.shape)
                    print(self.B.shape)
                    print(self.S_A.shape)
                    print(self.S_B.shape)
                    print("deleted ", k, ".te dimension")
            k = k + 1


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


def plott_data(y, z):
    x = np.arange(0., 500., 500 / len(y))

    plt.plot(x, y, 'b--', color='darkblue', label='Testerror')
    plt.plot(x, z, color='red', label='Trainerror')
    # plt.plot(x, z, 's -', color='blue')
    # plt.legend()
    plt.title("")
    plt.xlabel('Number of iteration')
    plt.ylabel('RMSE')
    plt.xlim([0, 10])
    plt.ylim([0, 2.5])
    plt.xticks([5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    plt.yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    # plt.grid(True)

    plt.legend()

    plt.show()


def generate_normal_data(mitarbeiter, feat_mitarbeiter, projekte, feat_projekte, feat_main, sparsity, binary):
    i = mitarbeiter
    j = projekte
    m = feat_mitarbeiter
    n = feat_projekte
    k = feat_main

    a_matrix = np.random.normal(0.0, 0.05, size=[m, k])
    f_matrix = np.random.randint(0, 6, size=[m, i])
    b_matrix = np.random.normal(0.0, 0.05, size=[n, k])
    if binary:
        g_matrix = np.random.randint(0, 2, size=[n, j])
    else:
        g_matrix = np.random.randint(0, 6, size=[n, j])

    e_u = np.random.normal(0.0, 0.3, size=[k, i])
    e_v = np.random.normal(0.0, 0.3, size=[k, j])

    u_matrix = a_matrix.T.dot(f_matrix) + e_u
    v_matrix = b_matrix.T.dot(g_matrix) + e_v

    e_x = np.random.normal(0.0, 1, size=[i, j])

    x = u_matrix.T.dot(v_matrix) + e_x

    x = sparse_maker(x, percentage=sparsity)

    return a_matrix, f_matrix, u_matrix, b_matrix, g_matrix, v_matrix, x, e_u


def sparse_maker(mat, percentage):
    mat_copy = mat.copy()
    set_to_zero = nonzeros(mat_copy, shuffle=True, percentage=percentage)
    mat_copy[set_to_zero] = 0.0
    return mat_copy


def split_data(mat, g, percentage, columnwise):
    if columnwise:
        mat_copy = mat.copy()
        g_copy = g.copy()
        takeout_col = int(mat.shape[1] * percentage / 100)
        testvectors_num = mat.shape[1] - takeout_col

        first_idx = np.random.randint(0, takeout_col - 1)
        second_idx = first_idx + testvectors_num

        if second_idx == mat.shape[1]:
            second_idx = second_idx - 1
        if first_idx == 0:
            first_idx = first_idx + 1

        print(first_idx, second_idx)

        test_data = mat_copy[:, first_idx:second_idx]
        train_data = np.concatenate((mat_copy[:, :first_idx], mat_copy[:, second_idx:]), axis=1)

        g_test = g_copy[:, first_idx:second_idx]
        g = np.concatenate((g_copy[:, :first_idx], g_copy[:, second_idx:]), axis=1)

    else:
        test_data = sparse_maker(mat, percentage=percentage)
        train_data = mat - test_data
        g_test = None

    return train_data, test_data, g, g_test


def nonzeros_tupel(matrix):
    nnz = np.nonzero(matrix)
    tup = list(zip(nnz[0], nnz[1]))
    return tup


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


def analyze_data(x, test_data, train_data):
    x_non_zeros = nonzeros(x, shuffle=False, percentage=None)
    train_data_nonzero_idx = nonzeros(train_data, shuffle=False, percentage=None)
    test_data_nonzero_idx = nonzeros(test_data, shuffle=False, percentage=None)
    test_data_ratings = test_data[test_data_nonzero_idx]
    train_data_ratings = train_data[train_data_nonzero_idx]

    print("Variance of data \t", np.var(x[x_non_zeros]), "\nMaximum:\t", np.max(x), "\nMinimum:\t",
          np.min(x[x_non_zeros]))
    print("................................................................")
    print("Variance of test data:\t", np.var(test_data[test_data_nonzero_idx]))
    print("Number of test Ratings:\t", len(test_data_nonzero_idx[0]))
    print(".................................................................")
    print("Variance of train data:", np.var(train_data[train_data_nonzero_idx]))
    print("Number of train Ratings:", len(train_data_nonzero_idx[0]))
    print(".................................................................")
    print("Average test ratings=", np.average(test_data_ratings))
    print("Average train ratings=", np.average(train_data_ratings))


def var_data(data):
    return np.var(data[nonzeros(data, shuffle=False, percentage=None)])


def print_ratings(predictions, test):
    print("\n\n\n")
    print("Num", "  Data", "\tPredictions", "\t     Diff")
    for i in range(len(test)):
        print(i, ")", test[i], "\t", predictions[i], "\t", np.abs(test[i] - predictions[i]))

    print("\n\n\n")
    error = math.sqrt(mean_squared_error(test, predictions))
    print(error)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    I = 100
    M = 30
    J = 100
    N = 20
    K = 8

    SPARSITY = 98
    COLUMNWISE = False
    BINARY = False
    DIM_REDUCTION = False
    MAIN_FEATURES = 14

    a, f, u, b, g, v, x, e_u = generate_normal_data(I, M, J, N, K, sparsity=SPARSITY, binary=BINARY)
    # x, g, f = generate_real_data(nonevalue=0.0)

    train_data, test_data, g_train, g_test = split_data(x, g, percentage=80, columnwise=COLUMNWISE)
    analyze_data(x, train_data, test_data)
    solution_instance = SVRS(train_data, test_data, g_train, f, main_features=MAIN_FEATURES, columnwise=COLUMNWISE,
                             dim_reduction=DIM_REDUCTION)
    print("First error: ", solution_instance.calc_error(train_data, g_test, testdata=False),
          solution_instance.calc_error(test_data, g_test, testdata=True))

    input("Press any Key to train the data:\n")

    solution_instance.train(50, g_test)

    predicted_ratings = solution_instance.U.T.dot(solution_instance.V)
    predictions = predicted_ratings[np.nonzero(test_data)]

    plot.master_plott(solution_instance.train_error, solution_instance.test_error, title="")
