import random
import numpy as np

def main():     #出力部
    x = [random.uniform(0, 100) for _ in range(100)]
    print("Ave:\t{}".format(calc_ave(x)))
    print("Var:\t{}".format(calc_var(x)))
    print("\nComputed by numpy")
    print("Ave:\t{}".format(np.mean(x)))
    print("Var:\t{}".format(np.var(x)))

def calc_ave(numbers):  #平均を求める関数
    return sum(numbers[i] for i in range(len(numbers))) / len(numbers)

def calc_var(numbers):  #分散を求める関数
    return sum(numbers[i]**2 for i in range(len(numbers))) / len(numbers) - calc_ave(numbers)**2

if __name__ == "__main__":
    main()