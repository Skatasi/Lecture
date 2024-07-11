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

    #散布図化
    plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    main()