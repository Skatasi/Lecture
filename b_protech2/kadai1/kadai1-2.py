a = float(input("Please a positive integer: "))
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

print("numGuesses = ", numGuesses) # 調べた回数
if abs(x**2 - a) >= epsilon:
    print("Failed on square root of ", a)
else:
    print(x, " is close to square root of ", a)