import pandas as pd
from matplotlib import pyplot as plt

#最新５０日分の日経平均株価を折れ線グラフとして出力するプログラム
def main():
    #csvファイルからの読み込み
    df = pd.read_csv("nikkei_stock_average_daily_jp.csv")

    #最新の日付から５０日分を取得
    df.sort_index(inplace=True)
    df = df.tail(50)

    #折れ線グラフ化
    df.plot(x= "Date")
    plt.show()

if __name__ == "__main__":
    main()