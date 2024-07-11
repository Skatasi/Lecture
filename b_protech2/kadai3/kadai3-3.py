import dist
import pandas as pd

def main(): #入出力部
    df = pd.read_table("locations.csv")
    capital1 = "Osaka City"
    capital2 = "Tokyo"
    lat1, lon1 = get_location(capital1, df)
    lat2, lon2 = get_location(capital2, df)

    distance = dist.calc_dist(lat1, lon1, lat2, lon2)

    print("{}:\t{:.4f}, {:.4f}".format(capital1, lat1, lon1))
    print("{}:\t{:.4f}, {:.4f}".format(capital2, lat2, lon2))
    print("Distance:\t{:.2f} km".format(distance))

def get_location(capital, df):  #データテーブルから指定した都市の緯度と経度を抽出する関数
    return float(df[df.capital_en==capital]["lat"]), float(df[df.capital_en==capital]["lon"])

if __name__ == "__main__":
    main()
