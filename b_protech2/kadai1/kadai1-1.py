a = float(input("Please a positive integer: "))
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

print("numGuesses = ", numGuesses) # 調べた回数
if abs(x**2 - a) >= epsilon:
    print("Failed on square root of ", a)
else:
    print(x, " is close to square root of ", a)