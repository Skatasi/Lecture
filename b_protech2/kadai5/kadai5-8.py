import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy

#100個の身長体重対データをランダムで抽出し、標準化するプログラム
def main():
    #csvファイルからの読み込み
    df = pd.read_csv("weight-height.csv")
    df_male = df[df["Gender"] == "Male"]
    df_female = df[df["Gender"] == "Female"]

    x_male = df_male["Weight"].tolist()
    x_female = df_female["Weight"].tolist()
    y_male = df_male["Height"].tolist()
    y_female = df_female["Height"].tolist()

    x_m_ave = np.mean(x_male)
    x_m_var = np.sqrt(np.var(x_male))
    y_m_ave = np.mean(y_male)
    y_m_var = np.sqrt(np.var(y_male))
    x_f_ave = np.mean(x_female)
    x_f_var = np.sqrt(np.var(x_female))
    y_f_ave = np.mean(y_female)
    y_f_var = np.sqrt(np.var(y_female))

    pvalue_x = scipy.stats.ttest_ind(x_male, x_female)[1]
    pvalue_y = scipy.stats.ttest_ind(y_male, y_female)[1]

    plt.figure(4).clf()
    plt.subplot(121)
    plt.bar([0, 1], [x_m_ave,x_f_ave], yerr=[x_m_var, x_f_var])
    plt.ylabel('Height')
    plt.xticks([0, 1], ['Male', 'Female'])
    
    plt.subplot(122)
    plt.bar([0, 1], [y_m_ave,y_f_ave], yerr=[y_m_var, y_f_var])
    plt.ylabel('Weight')
    plt.xticks([0, 1], ['Male', 'Female'])
    plt.tight_layout()
    
    if pvalue_x > .95:
        print('Female is higher than Male (p={:.3f})'.format(1 - pvalue_x))
    elif pvalue_x < .05:
        print('Male is higher than Female (p={:.3f})'.format(pvalue_x))
    else:
        print('There is no significant difference between Male and Female in height (p={:.3f})'.format(pvalue_x))
        
    if pvalue_y > .95:
        print('Female is heavier than Male (p={:.3f})'.format(1 - pvalue_y))
    elif pvalue_y < .05:
        print('Male is heavier than Female (p={:.3f})'.format(pvalue_y))
    else:
        print('There is no significant difference between Male and Female in weight (p={:.3f})'.format(pvalue_y))

if __name__ == "__main__":
    main()