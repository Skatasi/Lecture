import json
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import requests
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

videos_data = json.loads(open('output.json').read())

views = []
for video in videos_data:
    views.append(int(video['view_count']))

ave = sum(views) / len(views)
med = np.median(views)

# plt.hist(views, bins=20, color='skyblue')
# plt.axvline(ave, color='red', linestyle='dashed', linewidth=1)
# plt.axvline(med, color='green', linestyle='dashed', linewidth=1)
# plt.show()

def get_img(url):
    # URLにGET通信でアクセス
    res = requests.get(url)

   # 画像をファイルに保存
    with open('image.jpg', 'wb') as f:
        f.write(res.content)

def img_array(img):
    R = 0
    G = 0
    B = 0
    img_c = np.array(Image.open(img))
    img_g = np.array(Image.open(img).convert(mode="L"))
    for i in range(img_c.shape[0]):
        for j in range(img_c.shape[1]):
            R += img_c[i][j][0]
            G += img_c[i][j][1]
            B += img_c[i][j][2]
    return [R, G, B, img_g.mean()]
y = []
data = []
for video in videos_data:
    if int(video['view_count']) > ave:
        video['label'] = 1
    else:
        video['label'] = 0
    url = video['thumbnail']
    get_img(url)
    img = img_array("image.jpg")
    time = int(video['publish_time'][11:13])
    data.append([img[0], img[1], img[2], img[3], time])
    y.append(video['label'])

# PCAの実行
pca = PCA(n_components=2)
pca.fit(data)
data = np.array(data)
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# 主成分
principal_components = pca.components_
print("Principal Components:\n", principal_components)

# 主成分に基づいたデータの変換
data_pca = pca.transform(data)

# 主成分方向へのプロット
plt.scatter(data[:, 0], data[:, 1], c=y, edgecolors='k', marker='o', label='Original Data')
plt.quiver(0, 0, principal_components[0, 0], principal_components[0, 1], color='r', scale=5, label='1st Principal Component')
plt.quiver(0, 0, principal_components[1, 0], principal_components[1, 1], color='b', scale=5, label='2nd Principal Component')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PCA')
plt.show()
# SVDの実行
U, S, Vt = np.linalg.svd(data)

print("U matrix:\n", U)
print("Sigma values:\n", S)
print("Vt matrix:\n", Vt)

# SVDによる次元削減
data_svd = np.dot(U[:, :2], np.diag(S[:2]))

# SVDによる次元削減後のプロット
plt.scatter(data_svd[:, 0], data_svd[:, 1], c=y, edgecolors='k', marker='o', label='SVD Reduced Data')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('SVD')
plt.legend()
plt.show()

# X_train, X_test, y_train, y_test = train_test_split(data_svd, y, test_size=0.3)

# SVMモデルの作成
svm_model = SVC(kernel='linear')

# モデルの訓練
svm_model.fit(data_svd, y)

# 直線の係数（重みとバイアス）の取得
w = svm_model.coef_[0]
b = svm_model.intercept_[0]

print(f"Weights: {w}")
print(f"Bias: {b}")

# # テストデータで予測
# y_pred = svm_model.predict(X_test)

# # モデルの評価
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # 混同行列の表示
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# 決定境界のプロット
def plot_decision_boundary(model, X, y):
    h = .02  # メッシュのステップサイズ

    # カラーマップの設定
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # モデルの予測
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # データポイントと決定境界のプロット
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

# トレーニングデータの決定境界のプロット
plot_decision_boundary(svm_model, data_svd, y)