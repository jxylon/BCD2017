import NewMouseroad.read as br
import NewMouseroad.feature as bf
import math
import numpy as np
import NewMouseroad.read as br
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb

def x_y_t(df):
    # 得到x,y,t,target,label
    x, y, t, target, squ = [], [], [], [], []
    # Dataframe行遍历
    for l in range(len(df)):
        # 第0列：序号
        squ.append(df[0][l])
        # 第1列：坐标轨迹
        line = df[1][l].split(';')
        # 第2列：目标坐标
        target1 = (df[2][l])
        target1 = target1.split(',')
        target.append([target1[0], target1[1]])
        # 一次轨迹
        x1, y1, t1 = [], [], []
        # 拆分坐标轨迹
        for i in range(len(line) - 1):
            line1 = line[i].split(',')
            # x
            x1.append(line1[0])
            # y
            y1.append(line1[1])
            # time
            t1.append(line1[2])
        x.append(x1)
        y.append(y1)
        t.append(t1)
    return x, y, t, target, squ

# 写文件
def write(test_y, squence: list):
    fp = open('dsjtzs_txfzjh_preliminary.txt', 'w', encoding='utf-8')
    for i in range(len(test_y)):
        if test_y[i] <0.999:#current best 0.999
            fp.write(str(squence[i]) + '\n')
    fp.close()

if __name__ == '__main__':
    # 读取训练集
    df = br.read_file(2)
    # 得到x,y,t,label,target,squ
    x, y, t, label, target, squ = br.x_y_t(df)
    # 训练集特征
    train_x = np.array(bf.train(x, y, t,target))
    label = np.array(label)
    #参数
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
    lgb_train = lgb.Dataset(train_x, label)
    print('start training')
    gbm = lgb.train(params,lgb_train,num_boost_round=280)
    print('Done!')
    # 读取测试集
    print('start reading')
    df = br.read_file(3)
    print('Done!')
    # 得到x,y,t,target,squ
    print('start getting xyz')
    x, y, t, target, squ = x_y_t(df)
    print('Done!')
    # 测试集特征
    print('start getting test_features')
    test_x = np.array(bf.train(x, y, t,target))
    print('Done!')
    # 预测结果
    print('start predicting')
    test_y = gbm.predict(test_x)
    print('Done!')
    # 写文件
    print('start writing')
    write(test_y, squ)
    print('Done!')
