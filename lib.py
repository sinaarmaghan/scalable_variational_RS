import numpy as np


def calc_free_energy_mat_u(svrs: object) -> np.float64:
    gesamt_u_energy = 0.0
    for k in range(svrs.K):
        for i in range(svrs.I):
            gesamt_u_energy += -svrs.u_alpha_vector[k] / 2 * ((svrs.U[k, i] - np.dot(svrs.A[:, k], svrs.F[:, i])) ** 2
                                                              + svrs.S_U[k, i] +
                                                              np.dot(svrs.S_A[:, k], svrs.F[:, i] ** 2)) \
                               + 0.5 * np.log(svrs.S_U[k, i] * svrs.u_alpha_vector[k]) + 0.5
    return gesamt_u_energy


def calc_free_energy_mat_v(svrs: object) -> np.float64:
    gesamt_v_energy = 0.0

    for k in range(svrs.K):
        for j in range(svrs.J):
            gesamt_v_energy += calc_free_energy_v(svrs, k, j)

    return gesamt_v_energy


def calc_free_energy_mat_a(svrs: object) -> np.float64:
    gesamt_a_energy = 0.0
    for m in range(svrs.M):
        for k in range(svrs.K):
            gesamt_a_energy += calc_free_energy_a(svrs, m, k)

    return gesamt_a_energy


def calc_free_energy_mat_b(svrs: object) -> np.float64:
    gesamt_b_energy = 0.0

    for n in range(svrs.N):
        for k in range(svrs.K):
            gesamt_b_energy += calc_free_energy_b(svrs, n, k)
    return gesamt_b_energy


def calc_free_energy_mat_x(svrs: object) -> np.float64:
    gesamt_x_energy = 0.0

    for i in range(svrs.I):
        for j in range(svrs.J):
            if svrs.X[i, j] is not None:
                gesamt_x_energy += calc_free_energy_x(svrs, i, j)


def calculate_free_energy(svrs):
    x_energy = 0.0
    u_energy = 0.0
    v_energy = 0.0
    a_energy = 0.0
    b_energy = 0.0

    # Todo None should probably be changed by -1 or something similar
    for i in range(svrs.I):
        for j in range(svrs.J):
            if svrs.X[i, j] is not None:
                x_energy += calc_free_energy_x(svrs, i, j)

    for k in range(svrs.K):
        for i in range(svrs.I):
            u_energy += calc_free_energy_u(svrs, k, i)

    for k in range(svrs.K):
        for j in range(svrs.J):
            v_energy += calc_free_energy_v(svrs, k, j)

    for m in range(svrs.M):
        for k in range(svrs.K):
            a_energy += calc_free_energy_a(svrs, m, k)

    for n in range(svrs.N):
        for k in range(svrs.K):
            b_energy += calc_free_energy_b(svrs, n, k)

    return x_energy+u_energy+v_energy+a_energy+b_energy
    print_energy(x_energy, u_energy, v_energy, a_energy, b_energy)


def calc_free_energy_x(svrs, i, j):
    # Todo: if svrs.x_tau(division by zero)
    return -svrs.x_tau / 2 * (r_ij(svrs, i, j) ** 2 + w_ij(svrs, i, j)) - 0.5 * np.log(2 * np.pi * (1 / svrs.x_tau))


def r_ij(svrs, i, j):
    return svrs.X[i, j] - np.dot(svrs.U[:, i], svrs.V[:, j])


def w_ij(svrs, i, j):
    return (np.dot(svrs.U[:, i] ** 2, svrs.S_V[:, j]) + np.dot(svrs.V[:, j] ** 2, svrs.S_U[:, i])
            + np.dot(svrs.S_U[:, i], svrs.S_V[:, j]))


def calc_free_energy_u(svrs, k, i):
    return -svrs.u_alpha_vector[k] / 2 * ((svrs.U[k, i] - np.dot(svrs.A[:, k], svrs.F[:, i])) ** 2
                                          + svrs.S_U[k, i] +
                                          np.dot(svrs.S_A[:, k], svrs.F[:, i] ** 2)) \
           + 0.5 * np.log(svrs.S_U[k, i] * svrs.u_alpha_vector[k]) + 0.5


def calc_free_energy_v(svrs, k, j):
    return -svrs.v_beta_vector[k] / 2 * ((svrs.V[k, j] - np.dot(svrs.B[:, k], svrs.G[:, j])) ** 2
                                         + svrs.S_V[k, j] +
                                         np.dot(svrs.S_B[:, k], svrs.G[:, j] ** 2)) \
           + 0.5 * np.log(svrs.S_V[k, j] * svrs.v_beta_vector[k]) + 0.5


def calc_free_energy_a(svrs, m, k):
    return -svrs.a_phi_vector[k] / 2 * (svrs.A[m, k] ** 2 + svrs.S_A[m, k]) + \
           0.5 * np.log(svrs.S_A[m, k] * svrs.a_phi_vector[k]) + 0.5


def calc_free_energy_b(svrs, n, k):
    return -svrs.b_varphi_vector[k] / 2 * (svrs.B[n, k] ** 2 + svrs.S_B[n, k]) + \
           0.5 * np.log(svrs.S_B[n, k] * svrs.b_varphi_vector[k]) + 0.5

#csr
def init_omega(x, noneVal):
    m, n = x.shape
    omega_j = {}
    omega_i = {}
    omega_size = 0
    omega_size_test=0
    for i in range(m):
        omega_i_list = []
        for j in range(n):
            if x[i, j] != noneVal:
                omega_size= omega_size+1
                omega_i_list.append(j)

        omega_i.update({i: omega_i_list})

    for j in range(n):
        omega_j_list = []
        for i in range(m):
            if x[i, j] != noneVal:
                omega_size_test=omega_size_test+1
                omega_j_list.append(i)

        omega_j.update({j: omega_j_list})

    if omega_size_test != omega_size:
        print("iwas stimmt nicht:) ")
    return omega_i, omega_j ,omega_size

#Todo: generiere A,B F.G ----return X,F,G finde den Rest !
#
def generate_test_data():
    return None


def generate_test_data(mitarbeiter,  feat_mitarbeiter, projekte, feat_projekte, feat_main):
    i = mitarbeiter
    j = projekte
    m = feat_mitarbeiter
    n = feat_projekte
    k = feat_main

    amat = np.random.normal(0.0, 1.0, size=[m, k])
    fmat = np.random.normal(0.0, 1.0, size=[m, i])
    bmat = np.random.normal(0.0, 1.0, size=[n, k])
    gmat = np.random.normal(0.0, 1.0, size=[n, j])

    e_u = np.random.normal(0.0, 0.1, size=[k, i])
    e_v = np.random.normal(0.0, 0.1, size=[k, j])

    u = np.matmul(amat.transpose(), fmat) + e_u
    v = np.matmul(bmat.transpose(), gmat) + e_v

    e_x = np.random.normal(0.0, 0.1, size=[i, j])

    x = np.matmul(u.transpose(), v) + e_x

    return amat, fmat, u, bmat, gmat, v, x


def init_test_matrix(m, i, j, n , normal):


    if normal:
        fmat = np.random.normal(0.0, 1.0, size=[m, i])
        xmat = np.random.normal(0.0, 1.0, size=[i, j])
        gmat = np.random.normal(0.0, 1.0, size=[n, j])


    else:
        fmat = np.arange(m * i).reshape(m, i)
        xmat = np.arange(i * j).reshape(i, j)
        gmat = np.arange(n * j).reshape(n, j)

    return xmat, gmat, fmat


def print_energy(x, u, v, a, b):
    arguments = locals()
    sum = x + u + v + a + b

    print("Entire free Energy = ", sum)
    print("Free energy of X = ", x)
    print("Free energy of U = ", u)
    print("Free energy of V = ", v)
    print("Free energy of A = ", a)
    print("Free energy of B = ", b)

    # region old method calculate free energy
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
    #endregion