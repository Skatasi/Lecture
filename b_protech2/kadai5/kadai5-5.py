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

    #行列化
    X = []
    for _ in range(N):
        X.append([1,x[_]])
    X = np.matrix(X)
    Y = np.matrix([y]).T
    w = np.matrix([0,0]).T

    #回帰直線、MSEの導出
    mse = sum((Y[i,0] - np.dot(X[i],w)[0,0])**2 for i in range(N))/N
    mse_list = [mse]
    while 1:
        w = w - lr * np.dot(X.T,(np.dot(X,w)-Y))/N
        new_mse = sum((Y[i,0] - np.dot(X[i],w)[0,0])**2 for i in range(N))/N
        mse_list.append(new_mse)
        if (new_mse - mse)**2 < epsilon**2:
            break
        else:
            mse = new_mse

    #折れ線グラフ化
    print(mse_list)
    count = [i+1 for i in range(len(mse_list))]
    plt.scatter(count,mse_list)
    plt.show()


if __name__ == "__main__":
    main()