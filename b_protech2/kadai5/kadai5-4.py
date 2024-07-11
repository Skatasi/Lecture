import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
N=100
epsilon = 0.01
lr = 0.1

#100個の身長体重対データをランダムで抽出し、標準化するプログラム
def main():
    #csvファイルからの読み込み
    df = pd.read_csv("weight-height.csv")

    #100個をランダムに取得
    df = df.sample(N)
    x = df["Weight"].tolist()
    y = df["Height"].tolist()

    #標準化
    x = (x - np.mean(x))/np.sqrt(np.var(x))
    y = (y - np.mean(y))/np.sqrt(np.var(y))

    #回帰直線、MSEの導出
    a = 0
    b = 0
    mse = sum((y[i] - (a + b*x[i]))**2 for i in range(N))/N
    mse_list = [mse]
    while 1:
        new_a = a - lr * sum((a + b*x[i])- y[i] for i in range(N))/N
        new_b = b - lr * sum(x[i]*((a + b*x[i])- y[i]) for i in range(N))/N
        a = new_a
        b = new_b
        new_mse = sum((y[i] - (a + b*x[i]))**2 for i in range(N))/N
        mse_list.append(sum((y[i] - (a + b*x[i]))**2 for i in range(N))/N)
        if (new_mse - mse)**2 < epsilon**2:
            break
        else:
            mse = new_mse

    #折れ線グラフ化
    count = [i+1 for i in range(len(mse_list))]
    plt.scatter(count,mse_list)
    plt.show()


if __name__ == "__main__":
    main()