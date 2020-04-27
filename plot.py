from SVRS import *


def plott_data(y, z, title):
    x = np.arange(0., 200., 200 / len(y))

    plt.plot(x, y, 'b--', color='darkblue', label='Test')
    plt.plot(x, z, color='red', label='Train')
    # plt.plot(x, z, 's -', color='blue')
    # plt.legend()
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('RMSE')
    # plt.xlim([5, 500])
    # plt.ylim([0, 5])
    # plt.xticks([5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    # plt.yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    # plt.grid(True)

    plt.legend()

    plt.show()


def plott_datab(y, z, title):
    x = np.arange(0., 50., 50 / len(y))

    plt.plot(x, y, 'b--', color='darkblue', label='Testerr (splitted data)')
    plt.plot(x, z, color='darkred', label='Trainerr')
    # plt.plot(x, z, 's -', color='blue')
    # plt.legend()
    plt.title(title)
    plt.xlabel('Number of iteration')
    plt.ylabel('RMSE')
    plt.xlim([0, 30])
    plt.ylim([0, 50])
    plt.xticks([0, 5, 10, 15, 20, 25, 30])
    plt.yticks([1.0, 5, 10, 20, 30, 40, 50])
    # plt.grid(True)

    plt.legend()
    plt.show()


def master_plott(first_data, second_data, title):
    if len(first_data) != len(second_data):
        print("Bad shapes!", len(first_data), len(second_data))
        input("")

    y_limit = max(first_data)
    print(y_limit)

    x = np.arange(1., float(len(first_data) + 1), 1.0)

    plt.plot(x, first_data, color='m', label='Split test error')
    plt.plot(x, second_data, color='darkgreen', label='Test error')
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('RMSE')
    plt.xlim([1, len(first_data)])
    plt.ylim([0, int(y_limit)])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    obj_numb = 4
    itr_numb = 20

    train_err = np.zeros(shape=(obj_numb, itr_numb))
    test_err = np.zeros(shape=(obj_numb, itr_numb))
    split_test = np.zeros(shape=(obj_numb, itr_numb))
    split_train = np.zeros(shape=(obj_numb, itr_numb))
    test_variance = []
    train_variance = []

    I = 100
    M = 30
    J = 100
    N = 20
    K = 8
    BINARY = True
    x, g, f = generate_real_data(nonevalue=0.0)

    for i in range(obj_numb):

        split = not bool(i % 2)
        train_data, test_data, g_train, g_test = split_data(x, g, 80, split)
        test_variance.append(var_data(test_data))
        train_variance.append(var_data(train_data))
        svrsObj = SVRS(train_data, test_data, g_train, f, K, columnwise=split, dim_reduction=False)

        print("First error: ", svrsObj.calc_error(train_data, g_test, testdata=False),
              svrsObj.calc_error(test_data, g_test, testdata=True), "\n", g_test)

        svrsObj.train(itr_numb, g_test)

        ratingPred = svrsObj.U.T.dot(svrsObj.V)
        predictions = ratingPred[np.nonzero(test_data)]

        if split:
            split_test[int(i / 2), :] = svrsObj.test_error
            split_train[int(i / 2), :] = svrsObj.train_error

        else:
            test_err[i, :] = svrsObj.test_error
            train_err[i, :] = svrsObj.train_error

    avg_test_error = [np.average(a) for a in zip(*test_err)]
    avg_train_error = [np.average(a) for a in zip(*train_err)]
    avg_split_test = [np.average(a) for a in zip(*split_test)]
    avg_split_train = [np.average(a) for a in zip(*split_train)]
    master_plott(avg_split_test, avg_test_error, title="")

    print("-------------------------")
    print("minimum split test error:", min(avg_split_test), np.argmin(avg_split_test))
    print(avg_split_test)
    print("average test variance:", np.average(test_variance))
    print("average train variance:", np.average(train_variance))
    print("minimum test error:", min(avg_test_error), np.argmin(avg_test_error))
    print("minimum train error:", min(avg_train_error), np.argmin(avg_train_error))
    print(avg_test_error)
    print("-------------------------")
