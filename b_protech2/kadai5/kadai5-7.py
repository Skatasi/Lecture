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

    #回帰直線、MSEの導出
    a, b =np.polyfit(x, y, 1)
    mse = sum((y[i] - a*x[i] - b)**2 for i in range(N))/N

    #折れ線グラフ化
    print(mse)


if __name__ == "__main__":
    main()