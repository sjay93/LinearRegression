import pandas as pnd
import numpy as npy


def beta_calculation(input_x, input_y):
    dot_transpose_arr = input_x.transpose().dot(input_x)
    input_x_inverse = npy.linalg.pinv(dot_transpose_arr)
    input_x_transpose_y = input_x.transpose().dot(input_y)
    beta_cap = input_x_inverse.dot(input_x_transpose_y)
    return beta_cap


def calculate_intercept(inputx, inputy, bet):
    w_0 = npy.mean(inputy) - (
        npy.mean(inputx[0]) * bet[0] + npy.mean(inputx[1]) * bet[1] + npy.mean(inputx[2]) * bet[2] + npy.mean(
            inputx[3]) * bet[3])
    return w_0


def calculate_prediction(inputx, inter, be):
    predic = []
    for row in range(0,len(inputx)):
        inputx_0 = be[0] * inputx.iloc[[row],[0]]
        inputx_1 = be[1] * inputx.iloc[[row],[1]]
        inputx_2 = be[2] * inputx.iloc[[row],[2]]
        inputx_3 = be[3] * inputx.iloc[[row],[3]]
        inputx_total = inputx_0[0] + inputx_1[1] + inputx_2[2] + inputx_3[3]
        inputx_total.add(inter,fill_value=0)
        predic.append(int(round(inputx_total)))
    return predic


def calculate_accuracy(x,y, pr):
    count_of_mismatch = len(x)
    for val in range(0,len(x)):
        if (int(pr.__getitem__(val)) == int(y.iloc[val])):
            count_of_mismatch = count_of_mismatch - 1
    return count_of_mismatch


inputdata = pnd.read_csv('iris.data', delimiter=',', header=None)
for folds_count in range(10,31,5):
    inputdata = inputdata.sample(frac=1)
    accur = 0
    for numb in range(0,folds_count):
        training_data = []
        testing_data = []
        size_data = int(round(len(inputdata) / folds_count))
        testing_data = inputdata[numb*size_data:size_data+numb*size_data+1]
        training_data_first = inputdata[0:numb*size_data]
        training_data_last = inputdata[numb*size_data+size_data: len(inputdata)]
        sets = [training_data_first, training_data_last]
        training_data = pnd.concat(sets)
        test_x = testing_data[[0,1,2,3]]
        test_y = testing_data[[4]]
        train_x = training_data[[0,1,2,3]]
        train_y = training_data[[4]]
        test_y = test_y.replace({'Iris-setosa': 2, 'Iris-versicolor': 3, 'Iris-virginica': 4})
        train_y = train_y.replace({'Iris-setosa': 2, 'Iris-versicolor': 3, 'Iris-virginica': 4})
        beta = beta_calculation(train_x, train_y)
        intercept = calculate_intercept(train_x, train_y, beta)
        pred = calculate_prediction(test_x, intercept, beta)
        miss = calculate_accuracy(test_x, test_y, pred)
        accuracy = float((len(test_x) - miss) / float(len(test_x)) * 100.00)
        print 'Accuracy for',numb+1,'folds out of',folds_count,'folds is',accuracy
        accur = accur + accuracy
    print 'Average accuracy for',folds_count,'is',accur/folds_count
