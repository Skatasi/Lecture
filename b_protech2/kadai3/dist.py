import math

def main():
    #latは緯度、lonは経度。
    lat1, lon1 = 34.6937, 135.5022 #Osaka
    lat2, lon2 = 35.6896, 139.6921 #Tokyo
    print("Distance:\t{}".format(calc_dist(lat1, lon1, lat2, lon2)))

def calc_dist(x1, y1, x2, y2):
    x1 = x1 * math.pi / 180
    x2 = x2 * math.pi / 180
    y1 = y1 * math.pi / 180
    y2 = y2 * math.pi / 180
    return 2*6378.1*math.asin(math.sqrt( math.sin((x2-x1)/2)**2 + math.cos(x1)*math.cos(x2)*math.sin((y2-y1))**2 ))

if __name__ == "__main__":
    main()