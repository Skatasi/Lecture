a = float(input("Please a positive integer: "))
x = 1.0
epsilon = 0.01
numGuesses = 0
# x = 1.0 を中心に誤差がepsilon以下になるようにニュートン法で探査する。
while abs(x**2 - a)>=epsilon:
    func_x = x**2 - a
    funcd_x = 2*x
    x = x - func_x / funcd_x
    numGuesses += 1

print("numGuesses = ", numGuesses) # 調べた回数
if abs(x**2 - a) >= epsilon:
    print("Failed on square root of ", a)
else:
    print(x, " is close to square root of ", a)