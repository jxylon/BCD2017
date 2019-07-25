import math
import numpy as np
import NewMouseroad.read as br
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import train_test_split
import lightgbm as lgb

# 计算速度函数
def count_speed(start_x, start_y, end_x, end_y, start_time, end_time):
    x = math.pow(int(end_x) - int(start_x), 2)
    y = math.pow(int(end_y) - int(start_y), 2)
    if start_time == end_time:
        return '1'
    else:
        return math.sqrt(x + y) / (int(end_time) - int(start_time))


# 计算速度函数 返回list
def speed(x: list, y: list, t: list):
    speed = []
    if len(x) == 1:
        speed.append(0)
        return speed
    for i in range(len(x) - 1):
        if count_speed(x[i], y[i], x[i + 1], y[i + 1], t[i], t[i + 1]) == '1':
            continue
        else:
            v1 = count_speed(x[i], y[i], x[i + 1], y[i + 1], t[i], t[i + 1])
        speed.append(v1)
    if len(speed) == 0:
        speed.append(0)
    return speed


# 计算加速度函数
def count_acc(start_speed, end_speed, start_time, end_time):
    if start_time == end_time:
        return '1'
    else:
        return (end_speed - start_speed) / (end_time - start_time)


# 计算加速度函数 返回list
def acc(x: list, y: list, t: list):
    accleartion = []
    if len(x) == 1:
        accleartion.append(0)
        return accleartion
    v0 = 0
    for i in range(len(x) - 1):
        v1 = count_speed(x[i], y[i], x[i + 1], y[i + 1], t[i], t[i + 1])
        if count_acc(v0, v1, t[i], t[i + 1]) == '1':
            continue
        else:
            acc1 = count_acc(v0, v1, t[i], t[i + 1])
        v0 = v1
        accleartion.append(acc1)
    if len(accleartion) == 0:
        accleartion.append(0)
    return accleartion


# 方差
def var(a: list):
    s = sum(a) / len(a)
    f = 0
    for i in range(len(a)):
        f += math.pow(a[i] - s, 2)
    return f


# 特征处理
def mfeature(a: list):
    max1 = max(a)
    min1 = min(a)
    mean = sum(a) / len(a)
    diff = max1 - min1
    return max1, min1, mean, diff


neigh = KNeighborsClassifier(n_neighbors=3)


#  特征3
def f3(a: list, max_acc):
    if a[0] == 0:
        f1 = 0
    else:
        f1 = max_acc / a[0]
    if a[-1] == 0:
        f2 = 0
    else:
        f2 = max_acc / a[-1]
    return f1, f2


# 已知两向量求余弦值
def cosVector(x, y):
    result1 = 0.0
    result2 = 0.0
    result3 = 0.0
    for i in range(len(x)):
        result1 += x[i] * y[i]  # sum(X*Y)
        result2 += x[i] ** 2  # sum(X*X)
        result3 += y[i] ** 2  # sum(Y*Y)
        if ((result2 * result3) ** 0.5) == 0:
            result = 0
        else:
            result = (result1 / ((result2 * result3) ** 0.5))
    return result


# 角度
def ang(x: list, y: list):
    angle = []
    if len(x) <= 2:
        angle.append(-1.1)
    else:
        for i in range(1, len(x) - 1):
            a = [x[i - 1] - x[i], y[i - 1] - y[i]]
            b = [x[i + 1] - x[i], y[i + 1] - y[i]]
            f = cosVector(a, b)
            angle.append(f)
    return angle


# 相同坐标
def f6(x: list):
    n = 0
    same = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] == x[j] and (j not in same):
                n += 1
                same.append(j)
                break
    return n / len(x)


# 坐标回退
def back_num(a: list):
    x, y = 0, 0
    if len(a) == 1:
        return 0  # 1
    else:
        for i in range(len(a) - 1):
            if (a[i + 1] - a[i]) >= 0:
                x += 1
            else:
                y += 1
    if x >= y:
        return x
    else:
        return -y


def back_num1(a: list):
    x, y = 0, 0
    if len(a) == 1:
        return 1  # 1
    for i in range(len(a) - 1):
        if i == 0 and (a[i + 1] - a[i]) >= 0:
            t1 = 1
        if i == 0 and (a[i + 1] - a[i]) < 0:
            t1 = -1
        if (i != 0 and t1 == 1 and (a[i + 1] - a[i]) < 0) or (i != 0 and t1 == -1 and (a[i + 1] - a[i]) > 0):
            return 0
    return 1


# 到目标的距离
def gougu(x1, x2, y1, y2):
    x = (x1 - x2) ** 2
    y = (y1 - y2) ** 2
    return math.sqrt(x + y)


def distance(x: list, y: list, target: list):
    distance_list = []
    for i in range(len(x)):
        distance_list.append(gougu(x[i], target[0], y[i], target[1]))
    return distance_list


# 驻点
def zhudian(a: list):
    x = 0
    if (len(a) <= 2):
        return 0
    for i in range(len(a) - 2):
        if (a[i + 1] - a[i]) > 0 and a[i + 2] - a[i + 1] < 0:
            x += 1
        elif (a[i + 1] - a[i]) < 0 and a[i + 2] - a[i + 1] > 0:
            x += 1
    return x


# 得到特征
def train(x: list, y: list, t: list, target: list):
    train_x = []
    # train_y = label
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = int(x[i][j])
            y[i][j] = int(y[i][j])
            t[i][j] = int(t[i][j])
        target[i][0] = float(target[i][0])
        target[i][1] = float(target[i][1])
    for i in range(len(x)):
        # 加速度、速度列表
        acclist = acc(x[i], y[i], t[i])
        spelist = speed(x[i], y[i], t[i])
        angolist = ang(x[i], y[i])
        distancelist = distance(x[i], y[i], target[i])
        # 加速度特征，最大、最小、首尾差、平均值
        maxAcc, minAcc, meanAcc, diffAcc = mfeature(acclist)
        # 速度特征
        maxSpe, minSpe, meanSpe, diffSpe = mfeature(spelist)
        # 最大加速度除以首\尾加速度
        acc1, acc2 = f3(acclist, maxAcc)
        # x坐标特征
        max_x, min_x, mean_x, diff_x = mfeature(x[i])
        # y坐标特征
        max_y, min_y, mean_y, diff_y = mfeature(y[i])
        # 角度特征
        max_ag, min_ag, mean_ag, diff_ag = mfeature(angolist)
        # 距离目标
        max_ds, min_ds, mean_ds, var_ds = mfeature(distancelist)
        # 坐标相同的个数除以总数
        same_x, same_y = f6(x[i]), f6(y[i])
        # 坐标回退
        x_back, y_back = back_num1(x[i]), back_num1(y[i])
        # 驻点
        x_zd, y_zd = zhudian(x[i]), zhudian(y[i])
        # 特征集合
        # feature_importances [ 17  24  56  43  11 182  27  50 208  41  27 222  42  85 100  37 176   07 178  46 101]
        feature = [maxAcc, minAcc, diffAcc, meanAcc, maxSpe, minSpe, diffSpe, meanSpe, acc1, acc2, max_x, min_x, mean_x,
                   diff_x, max_y, min_y, mean_y, diff_y, x_back, y_back,max_ag, min_ag, mean_ag, diff_ag]
        # 加入train_x
        train_x.append(feature)
    # analog_score(train_x, train_y)
    return train_x


def train1(x: list, y: list, t: list, target: list, label: list):
    train_x = []
    train_y = label
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = int(x[i][j])
            y[i][j] = int(y[i][j])
            t[i][j] = int(t[i][j])
        target[i][0] = float(target[i][0])
        target[i][1] = float(target[i][1])
    for i in range(len(x)):
        # 加速度、速度列表
        acclist = acc(x[i], y[i], t[i])
        spelist = speed(x[i], y[i], t[i])
        angolist = ang(x[i], y[i])
        distancelist = distance(x[i], y[i], target[i])
        # 加速度特征，最大、最小、首尾差、平均值
        maxAcc, minAcc, meanAcc, varAcc = mfeature(acclist)
        # 速度特征
        maxSpe, minSpe, meanSpe, varSpe = mfeature(spelist)
        # 最大加速度除以首\尾加速度
        acc1, acc2 = f3(acclist, maxAcc)
        # x坐标特征
        max_x, min_x, mean_x, var_x = mfeature(x[i])
        # y坐标特征
        max_y, min_y, mean_y, var_y = mfeature(y[i])
        # 角度特征
        max_ag, min_ag, mean_ag, var_ag = mfeature(angolist)
        # 距离目标
        max_ds, min_ds, mean_ds, var_ds = mfeature(distancelist)
        # 坐标相同的个数除以总数
        same_x, same_y = f6(x[i]), f6(y[i])
        # 坐标回退
        x_back, y_back = back_num1(x[i]), back_num1(y[i])
        # 驻点
        x_zd, y_zd = zhudian(x[i]), zhudian(y[i])
        # 特征集合
        # feature_importances [ 17  24  56  43  11 182  27  50 208  41  27 222  42  85 100  37 176   07 178  46 101]
        feature = [maxAcc, minAcc, varAcc, meanAcc, maxSpe, minSpe, varSpe, meanSpe, acc1, acc2, max_x, min_x, mean_x,
                   max_y, min_y, mean_y, max_ag, min_ag, var_ag, mean_ag, x_back, y_back]
        # 加入train_x
        train_x.append(feature)
    analog_score(train_x, train_y)
    return train_x


# 模拟得分
def analog_score(train_x, train_y):
    # 将20%的训练集做验证集 80%做训练集
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    # 参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'num_leaves': 7,
        'learning_rate': 0.05,
        'feature_fraction': 0.83,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0
    }
    # 训练
    lgb_train = lgb.Dataset(train_x, train_y)
    print('start training')
    gbm = lgb.train(params, lgb_train, num_boost_round=280)
    print('Done!')
    # 预测
    predict = gbm.predict(test_x)
    # 分数
    TP = 0  # 将0预测为0
    FN = 0  # 将0预测为1
    FP = 0  # 将1预测为0
    TN = 0  # 将1预测为1
    for i in range(len(predict)):
        if test_y[i] < 0.999 and predict[i] < 0.999:
            TP += 1
        if test_y[i] < 0.999 and predict[i] >= 0.999:
            FN += 1
        if test_y[i] >= 0.999 and predict[i] < 0.999:
            FP += 1
        if test_y[i] >= 0.999 and predict[i] >= 0.999:
            TN += 1
    A = (TP + TN) / (TP + FN + FP + TN)
    P = TP / (TP + FP)  # 精确率
    R = TP / (TP + FN)  # 召回率
    F = ((5 * R * P / (2 * R + 3 * P)) * 100)  # 5PR/(3R+2P)*100
    print("P=%.2f%%,R=%.2f%%" % (P * 100, R * 100))
    print("得分:%.2f,准确率:%.2f" % (F, A))


if __name__ == '__main__':
    df = br.read_file(2)
    x, y, t, label, target, squ = br.x_y_t(df)
    train1(x, y, t, target, label)
    