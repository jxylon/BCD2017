# 读取文件内容
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain


def read_file(i):
    # 文件名
    fn = ['dsjtzs_txfz_training_sample', 'dsjtzs_txfz_test_sample', 'dsjtzs_txfz_training', 'dsjtzs_txfz_test1']
    index_name = ['a1', 'a2', 'a3', 'a4']
    # 读取文件
    df = pd.read_csv(fn[i] + '.txt', sep=' ', header=None)
    # 自定义列索引名称

    # 返回Dataframe
    return df


def x_y_t(df):
    # 得到x,y,t,target,label
    x, y, t, label, target, squ = [], [], [], [], [], []
    # Dataframe行遍历
    for l in range(len(df)):
        # 第0列：序号
        squ.append(df[0][l])
        # 第1列：坐标轨迹
        line = df[1][l].split(';')
        # 第2列：目标坐标
        target1=(df[2][l])
        target1=target1.split(',')
        target.append([target1[0],target1[1]])
        # 第3列：label
        label.append(df[3][l])
        # 一次轨迹
        x1, y1, t1= [], [], []
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
    return x, y, t, label, target, squ


def draw(x, y, t, label, squ):
    count = 0
    for i in chain(range(30,60), range(2940, 2970)):
        plt.subplot(6, 10, count + 1)
        plt.scatter(x[i], y[i], c=t[i], s=20)
        plt.xticks([])
        plt.yticks([])
        plt.title(str(squ[i]) + '    ' + str(label[i]))
        count += 1
    plt.show()


if __name__ == '__main__':
    df = read_file(2)
    x, y, t, label, target, squ = x_y_t(df)
    draw(x, y, t, label, squ)