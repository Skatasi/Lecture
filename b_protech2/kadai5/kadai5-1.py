import pandas as pd
import matplotlib.pyplot as plt
N=100

#100個の身長体重対データをランダムで抽出し、散布図にするプログラム
def main():
    #csvファイルからの読み込み
    df = pd.read_csv("weight-height.csv")

    #100個をランダムに取得
    df = df.sample(N)
    x = df["Weight"].tolist()
    y = df["Height"].tolist()

    #散布図化
    plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    main()