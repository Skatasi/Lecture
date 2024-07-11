def get_sqrt_by_nomal(a):
    x = 1.0
    epsilon = 0.01
    step = epsilon**2
    numGuesses = 0
    # x = 1.0からx**2とaの誤差がepsilon未満となるxをstep = 0.0001ごとに総当たりで調べる
    while abs(x**2 - a)>=epsilon:
        if x > a:
            x -= step
        else:
            x += step
        numGuesses += 1
        if abs(x**2 - a) <= epsilon:
            print("numGuesses = ", numGuesses) # 調べた回数
            print(x, " is close to square root of ", a)

def get_sqrt_by_bi(a):
    x = 1.0
    epsilon = 0.01
    step = abs(a-x)
    numGuesses = 0
    # x = 1.0 を中心に誤差がepsilon以下になるように二分探査する。
    while abs(x**2 - a)>=epsilon:
        step /= 2
        if x**2 > a:
            x -= step
        else:
            x += step
        numGuesses += 1
        if abs(x**2 - a) <= epsilon:
            print("numGuesses = ", numGuesses) # 調べた回数
            print(x, " is close to square root of ", a)
            return

def get_sqrt_by_newton(a):
    x = 1.0
    epsilon = 0.01
    numGuesses = 0
    # x = 1.0 を中心に誤差がepsilon以下になるように二分探査する。
    while abs(x**2 - a)>=epsilon:
        func_x = x**2 - a
        funcd_x = 2*x
        x = x - func_x / funcd_x
        numGuesses += 1
        if abs(x**2 - a) <= epsilon:
            print("numGuesses = ", numGuesses) # 調べた回数
            print(x, " is close to square root of ", a)
            return

def main():
    a = float(input("Please a positive integer: "))
    get_sqrt_by_nomal(a)
    get_sqrt_by_bi(a)
    get_sqrt_by_newton(a)

if __name__ == "__main__":
    main()