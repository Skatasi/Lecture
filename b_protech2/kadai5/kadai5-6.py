import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
N=100

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
    w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
    mse = sum((Y[i,0] - np.dot(X[i],w)[0,0])**2 for i in range(N))/N

    #折れ線グラフ化
    print(mse)


if __name__ == "__main__":
    main()