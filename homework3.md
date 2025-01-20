# 第三次作业

## 1. 证明 Proposition 2.5.1

解：给定问题中的目标函数是：
$$
\arg\max_w \left( \sum_{i=1}^{N} \log \left( \mathcal{N}(y_i | w_0 + w^\top x_i, \sigma^2) \right) + \sum_{j=1}^{D} \log \left( \mathcal{N}(w_j | 0, \tau^2) \right) \right)
$$
其中，$ \mathcal{N}(y_i | w_0 + w^\top x_i, \sigma^2) $ 表示 $ y_i $ 的高斯分布，其均值为 $ w_0 + w^\top x_i $，方差为 $ \sigma^2 $，并且 $ \mathcal{N}(w_j | 0, \tau^2) $ 表示权重 $ w_j $ 的高斯先验分布，其均值为 0，方差为 $ \tau^2 $。

这些高斯分布的对数概率如下：

对于每一个 $ y_i $，由于 $ y_i \sim \mathcal{N}(w_0 + w^\top x_i, \sigma^2) $，对数似然函数是：
$$
\log \mathcal{N}(y_i | w_0 + w^\top x_i, \sigma^2) = -\frac{1}{2} \log(2\pi\sigma^2) - \frac{(y_i - w_0 - w^\top x_i)^2}{2\sigma^2}
$$

对于每一个 $ w_j $，由于 $ w_j \sim \mathcal{N}(0, \tau^2) $，对数似然函数是：
$$
\log \mathcal{N}(w_j | 0, \tau^2) = -\frac{1}{2} \log(2\pi\tau^2) - \frac{w_j^2}{2\tau^2}
$$

将所有的对数似然函数加在一起，我们得到总的对数似然：
$$
\sum_{i=1}^{N} \log \left( \mathcal{N}(y_i | w_0 + w^\top x_i, \sigma^2) \right) = -\frac{N}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i - w_0 - w^\top x_i)^2
$$

$$
\sum_{j=1}^{D} \log \left( \mathcal{N}(w_j | 0, \tau^2) \right) = -\frac{D}{2} \log(2\pi\tau^2) - \frac{1}{2\tau^2} \sum_{j=1}^{D} w_j^2
$$

因此，总的对数似然函数是：
$$
\log P(y | w) + \log P(w) = -\frac{N}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i - w_0 - w^\top x_i)^2 - \frac{D}{2} \log(2\pi\tau^2) - \frac{1}{2\tau^2} \sum_{j=1}^{D} w_j^2
$$

为了最大化对数似然，我们实际上是在最小化负的对数似然函数。忽略常数项（与 $ w $ 无关的项），得到：
$$
-\log P(y | w) - \log P(w) \propto \frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i - w_0 - w^\top x_i)^2 + \frac{1}{2\tau^2} \sum_{j=1}^{D} w_j^2
$$
上面右式除以 $ N $ 得到：
$$
\frac{1}{2\sigma^2N} \sum_{i=1}^{N} (y_i - w_0 - w^\top x_i)^2 + \frac{1}{2\tau^2N} \sum_{j=1}^{D} w_j^2
$$
再乘 $ 2\sigma^2 $ 得到：
$$
\frac{1}{N} \sum_{i=1}^{N} (y_i - w_0 - w^\top x_i)^2 + \frac{\sigma^2}{\tau^2N} \sum_{j=1}^{D} w_j^2
$$
由于 $ \lambda = \frac{\sigma^2}{N \tau^2} $，替换可得：
$$
J(w) = \frac{1}{N} \sum_{i=1}^{N} (y_i - w^\top x_i)^2 + \lambda ||w||^2
$$
得证，最大化对数似然函数（1）与最小化岭回归损失函数 $ J(w) $ 是等价的。
## 编程题
### 1.基于降维的机器学习
数据集：kddcup99_Train.zip, kddcup99_Test.zip
数据描述：https://www.kdd.org/kdd-cup/view/kdd-cup-1999/Data，⽬的是通过 42 维的数据判断某⼀个数据包是否为 attack。
任务描述：选择 SVM + 任意 1 种 Classification 模型（具体模型不限，例如贝叶斯、决策树等，除了深度学习模型，可自由选择），结合两种降维方法，PCA 与 AutoEncoder，实现数据降维 + 分类。

要求输出：不同方法降低到不同维度对判别结果的影响（提升还是下降），Plot 出两种降维方法降到 2 维之后的结果（Normal 和 Attack 的 sample 用不同颜色）
解：代码如下
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from keras.models import Model
from keras.layers import Input, Dense

# 预处理
features = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type'
]

# 读取数据
train_set = pd.read_csv('kddcup99_train.csv', names=features, on_bad_lines='skip')
test_set = pd.read_csv('kddcup99_test.csv', names=features, on_bad_lines='skip')

# 打印第一行检查一下
print(train_set.head())

# 取样
train_set = train_set.sample(frac=0.01)
test_set = test_set.sample(frac=0.01)
print("Training set size:", train_set.shape)
print("Test set size:", test_set.shape)
print("Data loading completed")

# 标为'attack'
train_set['attack'] = train_set['attack_type'].apply(lambda x: 0 if x == 'normal.' else 1)
test_set['attack'] = test_set['attack_type'].apply(lambda x: 0 if x == 'normal.' else 1)

# 删除attack_type
train_set = train_set.drop(columns=['attack_type'])
test_set = test_set.drop(columns=['attack_type'])

# 合并
combined_set = pd.concat([train_set, test_set], axis=0, ignore_index=True)

combined_set_encoded = pd.get_dummies(combined_set, columns=['protocol_type', 'service', 'flag'])

# 分为训练集与测试集
train_set_encoded = combined_set_encoded.iloc[:train_set.shape[0], :]
test_set_encoded = combined_set_encoded.iloc[train_set.shape[0]:, :]

X_train_data = train_set_encoded.drop(columns=['attack'])
y_train_data = train_set_encoded['attack']
X_test_data = test_set_encoded.drop(columns=['attack'])
y_test_data = test_set_encoded['attack']

# 标准化
scaler = StandardScaler()
X_train_scaled_data = scaler.fit_transform(X_train_data)
X_test_scaled_data = scaler.transform(X_test_data)
print("Data standardization completed")

# PCA降维
print("Starting PCA for dimensionality reduction")
pca_model = PCA(n_components=2)  
X_train_pca_result = pca_model.fit_transform(X_train_scaled_data)
X_test_pca_result = pca_model.transform(X_test_scaled_data)

# PCA可视化
plt.scatter(X_train_pca_result[:, 0], X_train_pca_result[:, 1], c=y_train_data, cmap='coolwarm', s=2)
plt.title('PCA: Normal vs Attack')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Attack (1) or Normal (0)')
plt.show()
print("PCA dimensionality reduction finished")

# Autoencoder降维
print("Starting AutoEncoder for dimensionality reduction")
input_dim_data = X_train_scaled_data.shape[1]
encoding_dim_data = 2  

# AutoEncoder
input_layer_data = Input(shape=(input_dim_data,))
encoded_layer = Dense(encoding_dim_data, activation='relu')(input_layer_data)
decoded_layer = Dense(input_dim_data, activation='sigmoid')(encoded_layer)

autoencoder_model = Model(input_layer_data, decoded_layer)
encoder_model = Model(input_layer_data, encoded_layer)

autoencoder_model.compile(optimizer='adam', loss='mean_squared_error')
autoencoder_model.fit(X_train_scaled_data, X_train_scaled_data, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test_scaled_data, X_test_scaled_data))

# 得到AutoEncoder编码的数据
X_train_encoded_data = encoder_model.predict(X_train_scaled_data)
X_test_encoded_data = encoder_model.predict(X_test_scaled_data)

# AutoEncoder可视化
plt.scatter(X_train_encoded_data[:, 0], X_train_encoded_data[:, 1], c=y_train_data, cmap='coolwarm', s=2)
plt.title('AutoEncoder: Normal vs Attack')
plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.colorbar(label='Attack (1) or Normal (0)')
plt.show()
print("AutoEncoder dimensionality reduction finished")

# 使用SVM
svm_pca_model = SVC()
svm_pca_model.fit(X_train_pca_result, y_train_data)
y_pred_pca_result = svm_pca_model.predict(X_test_pca_result)
accuracy_pca_result = accuracy_score(y_test_data, y_pred_pca_result)
print(f'SVM (using PCA) classification accuracy: {accuracy_pca_result:.6f}')

svm_encoded_model = SVC()
svm_encoded_model.fit(X_train_encoded_data, y_train_data)
y_pred_encoded_result = svm_encoded_model.predict(X_test_encoded_data)
accuracy_encoded_result = accuracy_score(y_test_data, y_pred_encoded_result)
print(f'SVM (using AutoEncoder) classification accuracy: {accuracy_encoded_result:.6f}')

# 使用决策树
dt_pca_model = DecisionTreeClassifier()
dt_pca_model.fit(X_train_pca_result, y_train_data)
y_pred_pca_dt_result = dt_pca_model.predict(X_test_pca_result)
accuracy_pca_dt_result = accuracy_score(y_test_data, y_pred_pca_dt_result)
print(f"Decision Tree (using PCA) classification accuracy: {accuracy_pca_dt_result:.6f}")

dt_encoded_model = DecisionTreeClassifier()
dt_encoded_model.fit(X_train_encoded_data, y_train_data)
y_pred_encoded_dt_result = dt_encoded_model.predict(X_test_encoded_data)
accuracy_encoded_dt_result = accuracy_score(y_test_data, y_pred_encoded_dt_result)
print(f"Decision Tree (using AutoEncoder) classification accuracy: {accuracy_encoded_dt_result:.6f}")

```
```
PS D:\sa\experiment\math\3> & D:/miniconda/envs/py310/python.exe d:/sa/experiment/math/3/b1.py
   duration protocol_type service flag  src_bytes  ...  dst_host_serror_rate  dst_host_srv_serror_rate  dst_host_rerror_rate  dst_host_srv_rerror_rate  attack_type
0         0           tcp    http   SF        215  ...                   0.0                       0.0                   0.0                       0.0      normal.
1         0           tcp    http   SF        233  ...                   0.0                       0.0                   0.0                       0.0      normal.
2         0           tcp    http   SF        239  ...                   0.0                       0.0                   0.0                       0.0      normal.
3         0           tcp    http   SF        238  ...                   0.0                       0.0                   0.0                       0.0      normal.
4         0           tcp    http   SF        235  ...                   0.0                       0.0                   0.0                       0.0      normal.

[5 rows x 42 columns]
Training set size: (39198, 42)
Test set size: (9786, 42)
Data loading completed
Data standardization completed
Starting PCA for dimensionality reduction
PCA dimensionality reduction finished
Starting AutoEncoder for dimensionality reduction
2025-01-08 10:50:37.698125: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-01-08 10:50:37.699129: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Epoch 1/50
154/154 [==============================] - 1s 3ms/step - loss: 1.1460 - val_loss: 1.3828
Epoch 2/50
154/154 [==============================] - 0s 2ms/step - loss: 0.9934 - val_loss: 1.2727
Epoch 3/50
154/154 [==============================] - 0s 2ms/step - loss: 0.9363 - val_loss: 1.2452
Epoch 4/50
154/154 [==============================] - 0s 2ms/step - loss: 0.9121 - val_loss: 1.2272
Epoch 5/50
154/154 [==============================] - 0s 1ms/step - loss: 0.9024 - val_loss: 1.2213
Epoch 6/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8979 - val_loss: 1.2175
Epoch 7/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8946 - val_loss: 1.2145
Epoch 8/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8918 - val_loss: 1.2117
Epoch 9/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8892 - val_loss: 1.2092
Epoch 10/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8869 - val_loss: 1.2069
Epoch 11/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8847 - val_loss: 1.2048
Epoch 12/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8823 - val_loss: 1.2020
Epoch 13/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8800 - val_loss: 1.2000
Epoch 14/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8782 - val_loss: 1.1984
Epoch 15/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8768 - val_loss: 1.1970
Epoch 16/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8755 - val_loss: 1.1958
Epoch 17/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8744 - val_loss: 1.1946
Epoch 18/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8733 - val_loss: 1.1936
Epoch 19/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8723 - val_loss: 1.1926
Epoch 20/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8713 - val_loss: 1.1917
Epoch 21/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8705 - val_loss: 1.1908
Epoch 22/50
154/154 [==============================] - 0s 1ms/step - loss: 0.8697 - val_loss: 1.1900
Epoch 23/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8689 - val_loss: 1.1892
Epoch 24/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8682 - val_loss: 1.1885
Epoch 25/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8675 - val_loss: 1.1879
Epoch 26/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8669 - val_loss: 1.1873
Epoch 27/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8664 - val_loss: 1.1867
Epoch 28/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8658 - val_loss: 1.1862
Epoch 29/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8653 - val_loss: 1.1857
Epoch 30/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8649 - val_loss: 1.1853
Epoch 31/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8645 - val_loss: 1.1849
Epoch 32/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8641 - val_loss: 1.1845
Epoch 33/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8637 - val_loss: 1.1841
Epoch 34/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8633 - val_loss: 1.1837
Epoch 35/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8630 - val_loss: 1.1834
Epoch 36/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8627 - val_loss: 1.1831
Epoch 37/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8624 - val_loss: 1.1828
Epoch 38/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8621 - val_loss: 1.1826
Epoch 39/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8619 - val_loss: 1.1823
Epoch 40/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8616 - val_loss: 1.1821
Epoch 41/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8614 - val_loss: 1.1818
Epoch 42/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8612 - val_loss: 1.1816
Epoch 43/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8610 - val_loss: 1.1814
Epoch 44/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8608 - val_loss: 1.1812
Epoch 45/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8606 - val_loss: 1.1811
Epoch 46/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8604 - val_loss: 1.1809
Epoch 47/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8602 - val_loss: 1.1807
Epoch 48/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8601 - val_loss: 1.1806
Epoch 49/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8599 - val_loss: 1.1804
Epoch 50/50
154/154 [==============================] - 0s 2ms/step - loss: 0.8598 - val_loss: 1.1803
1225/1225 [==============================] - 1s 439us/step
306/306 [==============================] - 0s 476us/step
AutoEncoder dimensionality reduction finished
SVM (using PCA) classification accuracy: 0.997548
SVM (using AutoEncoder) classification accuracy: 0.977928
Decision Tree (using PCA) classification accuracy: 0.997650
Decision Tree (using AutoEncoder) classification accuracy: 0.985694
```
对于SVM方法，使用PCA降维准确率是0.997548，使用AutoEncoder准确率是0.977928；对于决策树方法，使用PCA降维准确率是0.997650，使用AutoEncoder准确率是0.985694。
PCA降维⽅法降到2维之后的结果如下图所示：
![PCA](1.png)
AutoEncoder降维⽅法降到2维之后的结果如下图所示：
![AutoEncoder](2.png)

### 2.深度学习训练
数据集：kddcup99_Train.zip, kddcup99_Test.zip
任务描述：实现一个简单的神经网络模型判别数据包是否为 attack，网络层数不小于 5 层，平台不限（TensorFlow、Pytorch 等都可），例如 42->36->24->12->6->1。尝试至少 2 种激活函数，至少 2 种 parameter initialization 方法，至少 2 种训练方法（SGD，SGD+Momentom，Adam），训练模型并判断训练结果。
要求输出：1）模型描述，层数，每一层参数，激活函数选择，loss 函数设置等；2）针对不同方法组合（至少 4 个组合），plot 出随着 epoch 增长 training error 和 test error 的变化情况。
解：代码如下：
``` python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 预处理
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type'
]

# 读取数据
train_data = pd.read_csv('kddcup99_train.csv', names=column_names, on_bad_lines='skip')
test_data = pd.read_csv('kddcup99_test.csv', names=column_names, on_bad_lines='skip')

# 取部分数据
train_data = train_data.sample(frac=0.01)
test_data = test_data.sample(frac=0.01)

# 重标记'attack'
for data in [train_data, test_data]:
    data['attack'] = data['attack_type'].apply(lambda x: 0 if x == 'normal.' else 1)
    data.drop(columns=['attack_type'], inplace=True)

# 合并
combined_data = pd.concat([train_data, test_data], ignore_index=True)
combined_data_encoded = pd.get_dummies(combined_data, columns=['protocol_type', 'service', 'flag'])

# 分成训练集与测试集
train_data_encoded = combined_data_encoded.iloc[:train_data.shape[0], :]
test_data_encoded = combined_data_encoded.iloc[train_data.shape[0]:, :]

X_train, y_train = train_data_encoded.drop(columns=['attack']), train_data_encoded['attack']
X_test, y_test = test_data_encoded.drop(columns=['attack']), test_data_encoded['attack']

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转化为 TensorFlow 张量
X_train_tensor = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# 建模
def create_model(input_size, activation, initializer):
    initializer_map = {
        'he_normal': tf.keras.initializers.HeNormal(),
        'glorot_uniform': tf.keras.initializers.GlorotUniform()
    }
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(36, activation=activation, kernel_initializer=initializer_map[initializer], input_shape=(input_size,)),
        tf.keras.layers.Dense(24, activation=activation, kernel_initializer=initializer_map[initializer]),
        tf.keras.layers.Dense(12, activation=activation, kernel_initializer=initializer_map[initializer]),
        tf.keras.layers.Dense(6, activation=activation, kernel_initializer=initializer_map[initializer]),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer_map[initializer])
    ])
    return model

# 训练
def train_model(model, X_train, y_train, X_test, y_test, optimizer, epochs=50, batch_size=256):
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)
    return history

# 可视化
def plot_loss(history, opt_name, init_name, activation_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title(f'Loss for model with {opt_name}, {init_name} initializer, {activation_name} activation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_{opt_name}_{init_name}_{activation_name}.png')

# 激活函数和优化器
activation_functions = {'ReLU': 'relu', 'Sigmoid': 'sigmoid'}
optimizers = {'SGD': tf.keras.optimizers.SGD(), 'Adam': tf.keras.optimizers.Adam()}
initializers = ['he_normal', 'glorot_uniform']

# 模型训练
for activation_name, activation in activation_functions.items():
    for opt_name, optimizer in optimizers.items():
        for init_name in initializers:
            print(f"\nTraining model with {opt_name} optimizer, {init_name} initializer, {activation_name} activation")
            model = create_model(X_train_scaled.shape[1], activation, init_name)
            history = train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, optimizer)
            plot_loss(history, opt_name, init_name, activation_name)

```

运行结果如下：
```
PS D:\sa\experiment\math\3> & D:/miniconda/envs/py310/python.exe d:/sa/experiment/math/3/b2.py
2025-01-09 19:54:17.289972: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-01-09 19:54:17.305060: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.

Training model with SGD optimizer, he_normal initializer, ReLU activation
D:\miniconda\envs\py310\lib\site-packages\keras\initializers\initializers_v2.py:120: UserWarning: The initializer HeNormal is unseeded and being called multiple times, which will return identical values  each time (even if the initializer is unseeded). Please update your code to provide a seed to the initializer, or avoid using the same initalizer instance more than once.
  warnings.warn(
Epoch 1/50
154/154 - 1s - loss: 0.2574 - accuracy: 0.9242 - val_loss: 0.1479 - val_accuracy: 0.9363 - 928ms/epoch - 6ms/step
Epoch 2/50
154/154 - 0s - loss: 0.0935 - accuracy: 0.9608 - val_loss: 0.1140 - val_accuracy: 0.9855 - 240ms/epoch - 2ms/step
Epoch 3/50
154/154 - 0s - loss: 0.0537 - accuracy: 0.9888 - val_loss: 0.1197 - val_accuracy: 0.9909 - 242ms/epoch - 2ms/step
Epoch 4/50
154/154 - 0s - loss: 0.0368 - accuracy: 0.9928 - val_loss: 0.1207 - val_accuracy: 0.9926 - 239ms/epoch - 2ms/step
Epoch 5/50
154/154 - 0s - loss: 0.0267 - accuracy: 0.9954 - val_loss: 0.1317 - val_accuracy: 0.9944 - 244ms/epoch - 2ms/step
Epoch 6/50
154/154 - 0s - loss: 0.0199 - accuracy: 0.9965 - val_loss: 0.1316 - val_accuracy: 0.9956 - 248ms/epoch - 2ms/step
Epoch 7/50
154/154 - 0s - loss: 0.0158 - accuracy: 0.9970 - val_loss: 0.1340 - val_accuracy: 0.9959 - 257ms/epoch - 2ms/step
Epoch 8/50
154/154 - 0s - loss: 0.0134 - accuracy: 0.9972 - val_loss: 0.1370 - val_accuracy: 0.9966 - 247ms/epoch - 2ms/step
Epoch 9/50
154/154 - 0s - loss: 0.0111 - accuracy: 0.9974 - val_loss: 0.1396 - val_accuracy: 0.9971 - 238ms/epoch - 2ms/step
Epoch 10/50
154/154 - 0s - loss: 0.0089 - accuracy: 0.9976 - val_loss: 0.1320 - val_accuracy: 0.9973 - 231ms/epoch - 1ms/step
Epoch 11/50
154/154 - 0s - loss: 0.0077 - accuracy: 0.9977 - val_loss: 0.1374 - val_accuracy: 0.9975 - 237ms/epoch - 2ms/step
Epoch 12/50
154/154 - 0s - loss: 0.0069 - accuracy: 0.9978 - val_loss: 0.1380 - val_accuracy: 0.9975 - 233ms/epoch - 2ms/step
Epoch 13/50
154/154 - 0s - loss: 0.0064 - accuracy: 0.9980 - val_loss: 0.1436 - val_accuracy: 0.9975 - 237ms/epoch - 2ms/step
Epoch 14/50
154/154 - 0s - loss: 0.0059 - accuracy: 0.9982 - val_loss: 0.1446 - val_accuracy: 0.9980 - 233ms/epoch - 2ms/step
Epoch 15/50
154/154 - 0s - loss: 0.0054 - accuracy: 0.9981 - val_loss: 0.1432 - val_accuracy: 0.9981 - 233ms/epoch - 2ms/step
Epoch 16/50
154/154 - 0s - loss: 0.0050 - accuracy: 0.9983 - val_loss: 0.1506 - val_accuracy: 0.9982 - 237ms/epoch - 2ms/step
Epoch 17/50
154/154 - 0s - loss: 0.0047 - accuracy: 0.9984 - val_loss: 0.1433 - val_accuracy: 0.9986 - 234ms/epoch - 2ms/step
Epoch 18/50
154/154 - 0s - loss: 0.0045 - accuracy: 0.9986 - val_loss: 0.1526 - val_accuracy: 0.9989 - 235ms/epoch - 2ms/step
Epoch 19/50
154/154 - 0s - loss: 0.0043 - accuracy: 0.9988 - val_loss: 0.1493 - val_accuracy: 0.9991 - 237ms/epoch - 2ms/step
Epoch 20/50
154/154 - 0s - loss: 0.0041 - accuracy: 0.9991 - val_loss: 0.1499 - val_accuracy: 0.9991 - 234ms/epoch - 2ms/step
Epoch 21/50
154/154 - 0s - loss: 0.0040 - accuracy: 0.9992 - val_loss: 0.1509 - val_accuracy: 0.9991 - 226ms/epoch - 1ms/step
Epoch 22/50
154/154 - 0s - loss: 0.0038 - accuracy: 0.9992 - val_loss: 0.1557 - val_accuracy: 0.9990 - 234ms/epoch - 2ms/step
Epoch 23/50
154/154 - 0s - loss: 0.0037 - accuracy: 0.9991 - val_loss: 0.1601 - val_accuracy: 0.9991 - 235ms/epoch - 2ms/step
Epoch 24/50
154/154 - 0s - loss: 0.0036 - accuracy: 0.9991 - val_loss: 0.1607 - val_accuracy: 0.9991 - 240ms/epoch - 2ms/step
Epoch 25/50
154/154 - 0s - loss: 0.0035 - accuracy: 0.9992 - val_loss: 0.1675 - val_accuracy: 0.9991 - 234ms/epoch - 2ms/step
Epoch 26/50
154/154 - 0s - loss: 0.0034 - accuracy: 0.9992 - val_loss: 0.1579 - val_accuracy: 0.9991 - 234ms/epoch - 2ms/step
Epoch 27/50
154/154 - 0s - loss: 0.0033 - accuracy: 0.9993 - val_loss: 0.1609 - val_accuracy: 0.9991 - 236ms/epoch - 2ms/step
Epoch 28/50
154/154 - 0s - loss: 0.0032 - accuracy: 0.9993 - val_loss: 0.1628 - val_accuracy: 0.9991 - 236ms/epoch - 2ms/step
Epoch 29/50
154/154 - 0s - loss: 0.0031 - accuracy: 0.9993 - val_loss: 0.1626 - val_accuracy: 0.9992 - 236ms/epoch - 2ms/step
Epoch 30/50
154/154 - 0s - loss: 0.0030 - accuracy: 0.9993 - val_loss: 0.1635 - val_accuracy: 0.9992 - 235ms/epoch - 2ms/step
Epoch 31/50
154/154 - 0s - loss: 0.0030 - accuracy: 0.9993 - val_loss: 0.1642 - val_accuracy: 0.9991 - 235ms/epoch - 2ms/step
Epoch 32/50
154/154 - 0s - loss: 0.0029 - accuracy: 0.9993 - val_loss: 0.1614 - val_accuracy: 0.9991 - 235ms/epoch - 2ms/step
Epoch 33/50
154/154 - 0s - loss: 0.0028 - accuracy: 0.9993 - val_loss: 0.1659 - val_accuracy: 0.9993 - 236ms/epoch - 2ms/step
Epoch 34/50
154/154 - 0s - loss: 0.0028 - accuracy: 0.9994 - val_loss: 0.1669 - val_accuracy: 0.9993 - 236ms/epoch - 2ms/step
Epoch 35/50
154/154 - 0s - loss: 0.0027 - accuracy: 0.9994 - val_loss: 0.1650 - val_accuracy: 0.9993 - 236ms/epoch - 2ms/step
Epoch 36/50
154/154 - 0s - loss: 0.0027 - accuracy: 0.9994 - val_loss: 0.1710 - val_accuracy: 0.9991 - 231ms/epoch - 2ms/step
Epoch 37/50
154/154 - 0s - loss: 0.0026 - accuracy: 0.9994 - val_loss: 0.1710 - val_accuracy: 0.9993 - 234ms/epoch - 2ms/step
Epoch 38/50
154/154 - 0s - loss: 0.0026 - accuracy: 0.9993 - val_loss: 0.1689 - val_accuracy: 0.9993 - 236ms/epoch - 2ms/step
Epoch 39/50
154/154 - 0s - loss: 0.0025 - accuracy: 0.9993 - val_loss: 0.1717 - val_accuracy: 0.9993 - 237ms/epoch - 2ms/step
Epoch 40/50
154/154 - 0s - loss: 0.0025 - accuracy: 0.9993 - val_loss: 0.1739 - val_accuracy: 0.9993 - 233ms/epoch - 2ms/step
Epoch 41/50
154/154 - 0s - loss: 0.0025 - accuracy: 0.9993 - val_loss: 0.1683 - val_accuracy: 0.9993 - 233ms/epoch - 2ms/step
Epoch 42/50
154/154 - 0s - loss: 0.0024 - accuracy: 0.9993 - val_loss: 0.1744 - val_accuracy: 0.9992 - 235ms/epoch - 2ms/step
Epoch 43/50
154/154 - 0s - loss: 0.0024 - accuracy: 0.9994 - val_loss: 0.1779 - val_accuracy: 0.9992 - 234ms/epoch - 2ms/step
Epoch 44/50
154/154 - 0s - loss: 0.0024 - accuracy: 0.9994 - val_loss: 0.1784 - val_accuracy: 0.9993 - 226ms/epoch - 1ms/step
Epoch 45/50
154/154 - 0s - loss: 0.0023 - accuracy: 0.9994 - val_loss: 0.1789 - val_accuracy: 0.9993 - 231ms/epoch - 2ms/step
Epoch 46/50
154/154 - 0s - loss: 0.0023 - accuracy: 0.9994 - val_loss: 0.1734 - val_accuracy: 0.9993 - 236ms/epoch - 2ms/step
Epoch 47/50
154/154 - 0s - loss: 0.0023 - accuracy: 0.9994 - val_loss: 0.1753 - val_accuracy: 0.9993 - 234ms/epoch - 2ms/step
Epoch 48/50
154/154 - 0s - loss: 0.0022 - accuracy: 0.9994 - val_loss: 0.1772 - val_accuracy: 0.9993 - 237ms/epoch - 2ms/step
Epoch 49/50
154/154 - 0s - loss: 0.0022 - accuracy: 0.9994 - val_loss: 0.1777 - val_accuracy: 0.9993 - 235ms/epoch - 2ms/step
Epoch 50/50
154/154 - 0s - loss: 0.0022 - accuracy: 0.9994 - val_loss: 0.1773 - val_accuracy: 0.9993 - 233ms/epoch - 2ms/step

Training model with SGD optimizer, glorot_uniform initializer, ReLU activation
D:\miniconda\envs\py310\lib\site-packages\keras\initializers\initializers_v2.py:120: UserWarning: The initializer GlorotUniform is unseeded and being called multiple times, which will return identical values  each time (even if the initializer is unseeded). Please update your code to provide a seed to the initializer, or avoid using the same initalizer instance more than once.
  warnings.warn(
Epoch 1/50
154/154 - 1s - loss: 0.3814 - accuracy: 0.9725 - val_loss: 0.2113 - val_accuracy: 0.9852 - 682ms/epoch - 4ms/step
Epoch 2/50
154/154 - 0s - loss: 0.1303 - accuracy: 0.9883 - val_loss: 0.1127 - val_accuracy: 0.9907 - 229ms/epoch - 1ms/step
Epoch 3/50
154/154 - 0s - loss: 0.0632 - accuracy: 0.9927 - val_loss: 0.0814 - val_accuracy: 0.9941 - 244ms/epoch - 2ms/step
Epoch 4/50
154/154 - 0s - loss: 0.0372 - accuracy: 0.9952 - val_loss: 0.0642 - val_accuracy: 0.9963 - 228ms/epoch - 1ms/step
Epoch 5/50
154/154 - 0s - loss: 0.0231 - accuracy: 0.9977 - val_loss: 0.0594 - val_accuracy: 0.9974 - 229ms/epoch - 1ms/step
Epoch 6/50
154/154 - 0s - loss: 0.0165 - accuracy: 0.9984 - val_loss: 0.0579 - val_accuracy: 0.9978 - 233ms/epoch - 2ms/step
Epoch 7/50
154/154 - 0s - loss: 0.0128 - accuracy: 0.9986 - val_loss: 0.0571 - val_accuracy: 0.9982 - 245ms/epoch - 2ms/step
Epoch 8/50
154/154 - 0s - loss: 0.0101 - accuracy: 0.9989 - val_loss: 0.0572 - val_accuracy: 0.9988 - 233ms/epoch - 2ms/step
Epoch 9/50
154/154 - 0s - loss: 0.0081 - accuracy: 0.9991 - val_loss: 0.0555 - val_accuracy: 0.9987 - 240ms/epoch - 2ms/step
Epoch 10/50
154/154 - 0s - loss: 0.0070 - accuracy: 0.9991 - val_loss: 0.0586 - val_accuracy: 0.9988 - 233ms/epoch - 2ms/step
Epoch 11/50
154/154 - 0s - loss: 0.0064 - accuracy: 0.9991 - val_loss: 0.0569 - val_accuracy: 0.9988 - 235ms/epoch - 2ms/step
Epoch 12/50
154/154 - 0s - loss: 0.0056 - accuracy: 0.9991 - val_loss: 0.0579 - val_accuracy: 0.9988 - 236ms/epoch - 2ms/step
Epoch 13/50
154/154 - 0s - loss: 0.0052 - accuracy: 0.9991 - val_loss: 0.0575 - val_accuracy: 0.9987 - 232ms/epoch - 2ms/step
Epoch 14/50
154/154 - 0s - loss: 0.0046 - accuracy: 0.9991 - val_loss: 0.0585 - val_accuracy: 0.9988 - 232ms/epoch - 2ms/step
Epoch 15/50
154/154 - 0s - loss: 0.0044 - accuracy: 0.9991 - val_loss: 0.0608 - val_accuracy: 0.9988 - 228ms/epoch - 1ms/step
Epoch 16/50
154/154 - 0s - loss: 0.0039 - accuracy: 0.9992 - val_loss: 0.0601 - val_accuracy: 0.9989 - 230ms/epoch - 1ms/step
Epoch 17/50
154/154 - 0s - loss: 0.0037 - accuracy: 0.9993 - val_loss: 0.0651 - val_accuracy: 0.9989 - 228ms/epoch - 1ms/step
Epoch 18/50
154/154 - 0s - loss: 0.0036 - accuracy: 0.9993 - val_loss: 0.0650 - val_accuracy: 0.9989 - 232ms/epoch - 2ms/step
Epoch 19/50
154/154 - 0s - loss: 0.0033 - accuracy: 0.9993 - val_loss: 0.0691 - val_accuracy: 0.9990 - 229ms/epoch - 1ms/step
Epoch 20/50
154/154 - 0s - loss: 0.0034 - accuracy: 0.9993 - val_loss: 0.0705 - val_accuracy: 0.9990 - 227ms/epoch - 1ms/step
Epoch 21/50
154/154 - 0s - loss: 0.0032 - accuracy: 0.9992 - val_loss: 0.0751 - val_accuracy: 0.9989 - 226ms/epoch - 1ms/step
Epoch 22/50
154/154 - 0s - loss: 0.0032 - accuracy: 0.9993 - val_loss: 0.0744 - val_accuracy: 0.9989 - 232ms/epoch - 2ms/step
Epoch 23/50
154/154 - 0s - loss: 0.0031 - accuracy: 0.9993 - val_loss: 0.0747 - val_accuracy: 0.9990 - 229ms/epoch - 1ms/step
Epoch 24/50
154/154 - 0s - loss: 0.0030 - accuracy: 0.9993 - val_loss: 0.0791 - val_accuracy: 0.9989 - 231ms/epoch - 2ms/step
Epoch 25/50
154/154 - 0s - loss: 0.0031 - accuracy: 0.9993 - val_loss: 0.0775 - val_accuracy: 0.9989 - 234ms/epoch - 2ms/step
Epoch 26/50
154/154 - 0s - loss: 0.0029 - accuracy: 0.9993 - val_loss: 0.0795 - val_accuracy: 0.9989 - 229ms/epoch - 1ms/step
Epoch 27/50
154/154 - 0s - loss: 0.0030 - accuracy: 0.9993 - val_loss: 0.0810 - val_accuracy: 0.9990 - 234ms/epoch - 2ms/step
Epoch 28/50
154/154 - 0s - loss: 0.0028 - accuracy: 0.9993 - val_loss: 0.0804 - val_accuracy: 0.9991 - 232ms/epoch - 2ms/step
Epoch 29/50
154/154 - 0s - loss: 0.0027 - accuracy: 0.9993 - val_loss: 0.0849 - val_accuracy: 0.9991 - 231ms/epoch - 1ms/step
Epoch 30/50
154/154 - 0s - loss: 0.0028 - accuracy: 0.9993 - val_loss: 0.0829 - val_accuracy: 0.9991 - 231ms/epoch - 1ms/step
Epoch 31/50
154/154 - 0s - loss: 0.0027 - accuracy: 0.9992 - val_loss: 0.0842 - val_accuracy: 0.9991 - 229ms/epoch - 1ms/step
Epoch 32/50
154/154 - 0s - loss: 0.0027 - accuracy: 0.9993 - val_loss: 0.0838 - val_accuracy: 0.9991 - 232ms/epoch - 2ms/step
Epoch 33/50
154/154 - 0s - loss: 0.0025 - accuracy: 0.9994 - val_loss: 0.0838 - val_accuracy: 0.9992 - 237ms/epoch - 2ms/step
Epoch 34/50
154/154 - 0s - loss: 0.0025 - accuracy: 0.9993 - val_loss: 0.0878 - val_accuracy: 0.9991 - 230ms/epoch - 1ms/step
Epoch 35/50
154/154 - 0s - loss: 0.0026 - accuracy: 0.9994 - val_loss: 0.0837 - val_accuracy: 0.9990 - 228ms/epoch - 1ms/step
Epoch 36/50
154/154 - 0s - loss: 0.0024 - accuracy: 0.9994 - val_loss: 0.0837 - val_accuracy: 0.9992 - 231ms/epoch - 1ms/step
Epoch 37/50
154/154 - 0s - loss: 0.0024 - accuracy: 0.9994 - val_loss: 0.0847 - val_accuracy: 0.9992 - 232ms/epoch - 2ms/step
Epoch 38/50
154/154 - 0s - loss: 0.0024 - accuracy: 0.9994 - val_loss: 0.0848 - val_accuracy: 0.9993 - 231ms/epoch - 2ms/step
Epoch 39/50
154/154 - 0s - loss: 0.0024 - accuracy: 0.9994 - val_loss: 0.0840 - val_accuracy: 0.9992 - 229ms/epoch - 1ms/step
Epoch 40/50
154/154 - 0s - loss: 0.0023 - accuracy: 0.9995 - val_loss: 0.0869 - val_accuracy: 0.9992 - 228ms/epoch - 1ms/step
Epoch 41/50
154/154 - 0s - loss: 0.0023 - accuracy: 0.9994 - val_loss: 0.0877 - val_accuracy: 0.9993 - 230ms/epoch - 1ms/step
Epoch 42/50
154/154 - 0s - loss: 0.0022 - accuracy: 0.9995 - val_loss: 0.0869 - val_accuracy: 0.9993 - 236ms/epoch - 2ms/step
Epoch 43/50
154/154 - 0s - loss: 0.0022 - accuracy: 0.9994 - val_loss: 0.0869 - val_accuracy: 0.9993 - 229ms/epoch - 1ms/step
Epoch 44/50
154/154 - 0s - loss: 0.0022 - accuracy: 0.9994 - val_loss: 0.0899 - val_accuracy: 0.9991 - 226ms/epoch - 1ms/step
Epoch 45/50
154/154 - 0s - loss: 0.0023 - accuracy: 0.9995 - val_loss: 0.0869 - val_accuracy: 0.9992 - 227ms/epoch - 1ms/step
Epoch 46/50
154/154 - 0s - loss: 0.0022 - accuracy: 0.9995 - val_loss: 0.0871 - val_accuracy: 0.9993 - 231ms/epoch - 1ms/step
Epoch 47/50
154/154 - 0s - loss: 0.0023 - accuracy: 0.9995 - val_loss: 0.0892 - val_accuracy: 0.9993 - 246ms/epoch - 2ms/step
Epoch 48/50
154/154 - 0s - loss: 0.0021 - accuracy: 0.9995 - val_loss: 0.0859 - val_accuracy: 0.9993 - 234ms/epoch - 2ms/step
Epoch 49/50
154/154 - 0s - loss: 0.0021 - accuracy: 0.9995 - val_loss: 0.0894 - val_accuracy: 0.9992 - 229ms/epoch - 1ms/step
Epoch 50/50
154/154 - 0s - loss: 0.0023 - accuracy: 0.9995 - val_loss: 0.0881 - val_accuracy: 0.9991 - 236ms/epoch - 2ms/step

Training model with Adam optimizer, he_normal initializer, ReLU activation
Epoch 1/50
154/154 - 1s - loss: 0.2894 - accuracy: 0.9700 - val_loss: 0.0768 - val_accuracy: 0.9968 - 814ms/epoch - 5ms/step
Epoch 2/50
154/154 - 0s - loss: 0.0112 - accuracy: 0.9981 - val_loss: 0.0821 - val_accuracy: 0.9986 - 257ms/epoch - 2ms/step
Epoch 3/50
154/154 - 0s - loss: 0.0049 - accuracy: 0.9991 - val_loss: 0.0856 - val_accuracy: 0.9987 - 255ms/epoch - 2ms/step
Epoch 4/50
154/154 - 0s - loss: 0.0033 - accuracy: 0.9992 - val_loss: 0.0873 - val_accuracy: 0.9988 - 254ms/epoch - 2ms/step
Epoch 5/50
154/154 - 0s - loss: 0.0021 - accuracy: 0.9994 - val_loss: 0.0726 - val_accuracy: 0.9990 - 247ms/epoch - 2ms/step
Epoch 6/50
154/154 - 0s - loss: 0.0019 - accuracy: 0.9994 - val_loss: 0.0781 - val_accuracy: 0.9985 - 257ms/epoch - 2ms/step
Epoch 8/50
154/154 - 0s - loss: 0.0014 - accuracy: 0.9996 - val_loss: 0.0920 - val_accuracy: 0.9991 - 267ms/epoch - 2ms/step
Epoch 9/50
154/154 - 0s - loss: 0.0017 - accuracy: 0.9996 - val_loss: 0.0969 - val_accuracy: 0.9990 - 266ms/epoch - 2ms/step
Epoch 10/50
154/154 - 0s - loss: 0.0011 - accuracy: 0.9996 - val_loss: 0.0676 - val_accuracy: 0.9991 - 265ms/epoch - 2ms/step
Epoch 11/50
154/154 - 0s - loss: 0.0010 - accuracy: 0.9996 - val_loss: 0.0572 - val_accuracy: 0.9991 - 260ms/epoch - 2ms/step
Epoch 12/50
154/154 - 0s - loss: 8.5770e-04 - accuracy: 0.9996 - val_loss: 0.0834 - val_accuracy: 0.9990 - 255ms/epoch - 2ms/step
Epoch 13/50
154/154 - 0s - loss: 0.0010 - accuracy: 0.9997 - val_loss: 0.0841 - val_accuracy: 0.9991 - 257ms/epoch - 2ms/step
Epoch 14/50
154/154 - 0s - loss: 6.1481e-04 - accuracy: 0.9997 - val_loss: 0.0780 - val_accuracy: 0.9991 - 252ms/epoch - 2ms/step
Epoch 15/50
154/154 - 0s - loss: 5.4998e-04 - accuracy: 0.9998 - val_loss: 0.1051 - val_accuracy: 0.9991 - 259ms/epoch - 2ms/step
Epoch 16/50
154/154 - 0s - loss: 7.6684e-04 - accuracy: 0.9998 - val_loss: 0.0711 - val_accuracy: 0.9991 - 285ms/epoch - 2ms/step
Epoch 17/50
154/154 - 0s - loss: 7.7453e-04 - accuracy: 0.9998 - val_loss: 0.0885 - val_accuracy: 0.9990 - 334ms/epoch - 2ms/step
Epoch 18/50
154/154 - 0s - loss: 4.5202e-04 - accuracy: 0.9998 - val_loss: 0.0914 - val_accuracy: 0.9992 - 310ms/epoch - 2ms/step
Epoch 19/50
154/154 - 0s - loss: 4.9258e-04 - accuracy: 0.9998 - val_loss: 0.0807 - val_accuracy: 0.9992 - 258ms/epoch - 2ms/step
Epoch 20/50
154/154 - 0s - loss: 4.3710e-04 - accuracy: 0.9999 - val_loss: 0.0980 - val_accuracy: 0.9992 - 263ms/epoch - 2ms/step
Epoch 21/50
154/154 - 0s - loss: 3.0808e-04 - accuracy: 0.9999 - val_loss: 0.1015 - val_accuracy: 0.9992 - 263ms/epoch - 2ms/step
Epoch 22/50
154/154 - 0s - loss: 3.6415e-04 - accuracy: 0.9999 - val_loss: 0.1105 - val_accuracy: 0.9993 - 261ms/epoch - 2ms/step
Epoch 23/50
154/154 - 0s - loss: 3.4119e-04 - accuracy: 0.9999 - val_loss: 0.1044 - val_accuracy: 0.9992 - 259ms/epoch - 2ms/step
Epoch 24/50
154/154 - 0s - loss: 2.9447e-04 - accuracy: 0.9999 - val_loss: 0.0584 - val_accuracy: 0.9992 - 253ms/epoch - 2ms/step
Epoch 25/50
154/154 - 0s - loss: 4.2491e-04 - accuracy: 0.9998 - val_loss: 0.1267 - val_accuracy: 0.9992 - 257ms/epoch - 2ms/step
Epoch 26/50
154/154 - 0s - loss: 3.3274e-04 - accuracy: 0.9999 - val_loss: 0.1227 - val_accuracy: 0.9992 - 258ms/epoch - 2ms/step
Epoch 27/50
154/154 - 0s - loss: 2.6028e-04 - accuracy: 0.9999 - val_loss: 0.1015 - val_accuracy: 0.9993 - 265ms/epoch - 2ms/step
Epoch 28/50
154/154 - 0s - loss: 3.1660e-04 - accuracy: 0.9998 - val_loss: 0.0920 - val_accuracy: 0.9993 - 252ms/epoch - 2ms/step
Epoch 29/50
154/154 - 0s - loss: 8.2513e-04 - accuracy: 0.9998 - val_loss: 0.0791 - val_accuracy: 0.9991 - 256ms/epoch - 2ms/step
Epoch 30/50
154/154 - 0s - loss: 0.0018 - accuracy: 0.9996 - val_loss: 0.1162 - val_accuracy: 0.9990 - 252ms/epoch - 2ms/step
Epoch 31/50
154/154 - 0s - loss: 0.0014 - accuracy: 0.9998 - val_loss: 0.0997 - val_accuracy: 0.9992 - 254ms/epoch - 2ms/step
Epoch 32/50
154/154 - 0s - loss: 3.5293e-04 - accuracy: 0.9999 - val_loss: 0.1124 - val_accuracy: 0.9992 - 260ms/epoch - 2ms/step
Epoch 33/50
154/154 - 0s - loss: 2.6791e-04 - accuracy: 0.9999 - val_loss: 0.1164 - val_accuracy: 0.9992 - 252ms/epoch - 2ms/step
Epoch 34/50
154/154 - 0s - loss: 3.0358e-04 - accuracy: 0.9999 - val_loss: 0.1181 - val_accuracy: 0.9993 - 263ms/epoch - 2ms/step
Epoch 35/50
154/154 - 0s - loss: 2.2534e-04 - accuracy: 0.9999 - val_loss: 0.1348 - val_accuracy: 0.9993 - 252ms/epoch - 2ms/step
Epoch 36/50
154/154 - 0s - loss: 2.6585e-04 - accuracy: 0.9999 - val_loss: 0.1274 - val_accuracy: 0.9992 - 251ms/epoch - 2ms/step
Epoch 37/50
154/154 - 0s - loss: 3.4116e-04 - accuracy: 0.9999 - val_loss: 0.1336 - val_accuracy: 0.9993 - 257ms/epoch - 2ms/step
Epoch 38/50
154/154 - 0s - loss: 2.3273e-04 - accuracy: 0.9999 - val_loss: 0.1378 - val_accuracy: 0.9993 - 253ms/epoch - 2ms/step
Epoch 39/50
154/154 - 0s - loss: 2.4560e-04 - accuracy: 0.9999 - val_loss: 0.1435 - val_accuracy: 0.9993 - 256ms/epoch - 2ms/step
Epoch 40/50
154/154 - 0s - loss: 2.0030e-04 - accuracy: 0.9999 - val_loss: 0.1519 - val_accuracy: 0.9993 - 259ms/epoch - 2ms/step
Epoch 41/50
154/154 - 0s - loss: 2.4143e-04 - accuracy: 0.9999 - val_loss: 0.1423 - val_accuracy: 0.9993 - 256ms/epoch - 2ms/step
Epoch 42/50
154/154 - 0s - loss: 2.4981e-04 - accuracy: 0.9999 - val_loss: 0.1629 - val_accuracy: 0.9993 - 254ms/epoch - 2ms/step
Epoch 43/50
154/154 - 0s - loss: 2.2546e-04 - accuracy: 0.9999 - val_loss: 0.1540 - val_accuracy: 0.9993 - 252ms/epoch - 2ms/step
Epoch 44/50
154/154 - 0s - loss: 2.2822e-04 - accuracy: 0.9999 - val_loss: 0.1579 - val_accuracy: 0.9993 - 254ms/epoch - 2ms/step
Epoch 45/50
154/154 - 0s - loss: 2.8865e-04 - accuracy: 0.9998 - val_loss: 0.1663 - val_accuracy: 0.9993 - 255ms/epoch - 2ms/step
Epoch 46/50
154/154 - 0s - loss: 2.2648e-04 - accuracy: 0.9999 - val_loss: 0.1639 - val_accuracy: 0.9994 - 251ms/epoch - 2ms/step
Epoch 47/50
154/154 - 0s - loss: 1.8969e-04 - accuracy: 0.9999 - val_loss: 0.1611 - val_accuracy: 0.9992 - 256ms/epoch - 2ms/step
Epoch 48/50
154/154 - 0s - loss: 2.3252e-04 - accuracy: 0.9999 - val_loss: 0.1746 - val_accuracy: 0.9993 - 266ms/epoch - 2ms/step
Epoch 49/50
154/154 - 0s - loss: 1.5977e-04 - accuracy: 1.0000 - val_loss: 0.1747 - val_accuracy: 0.9994 - 256ms/epoch - 2ms/step
Epoch 50/50
154/154 - 0s - loss: 2.0760e-04 - accuracy: 0.9999 - val_loss: 0.1761 - val_accuracy: 0.9994 - 256ms/epoch - 2ms/step

Training model with Adam optimizer, glorot_uniform initializer, ReLU activation
Epoch 1/50
154/154 - 1s - loss: 0.0484 - accuracy: 0.9824 - val_loss: 0.1385 - val_accuracy: 0.9989 - 781ms/epoch - 5ms/step
Epoch 2/50
154/154 - 0s - loss: 0.0033 - accuracy: 0.9991 - val_loss: 0.1227 - val_accuracy: 0.9990 - 253ms/epoch - 2ms/step
Epoch 3/50
154/154 - 0s - loss: 0.0023 - accuracy: 0.9995 - val_loss: 0.1046 - val_accuracy: 0.9993 - 255ms/epoch - 2ms/step
Epoch 4/50
154/154 - 0s - loss: 0.0025 - accuracy: 0.9994 - val_loss: 0.1047 - val_accuracy: 0.9993 - 245ms/epoch - 2ms/step
Epoch 5/50
154/154 - 0s - loss: 0.0016 - accuracy: 0.9994 - val_loss: 0.1298 - val_accuracy: 0.9992 - 250ms/epoch - 2ms/step
Epoch 6/50
154/154 - 0s - loss: 0.0017 - accuracy: 0.9995 - val_loss: 0.1052 - val_accuracy: 0.9993 - 250ms/epoch - 2ms/step
Epoch 7/50
154/154 - 0s - loss: 0.0010 - accuracy: 0.9997 - val_loss: 0.0962 - val_accuracy: 0.9993 - 246ms/epoch - 2ms/step
Epoch 8/50
154/154 - 0s - loss: 0.0010 - accuracy: 0.9997 - val_loss: 0.1144 - val_accuracy: 0.9993 - 253ms/epoch - 2ms/step
Epoch 9/50
154/154 - 0s - loss: 8.9800e-04 - accuracy: 0.9997 - val_loss: 0.0942 - val_accuracy: 0.9993 - 250ms/epoch - 2ms/step
Epoch 10/50
154/154 - 0s - loss: 0.0012 - accuracy: 0.9997 - val_loss: 0.0838 - val_accuracy: 0.9994 - 248ms/epoch - 2ms/step
Epoch 11/50
154/154 - 0s - loss: 7.6047e-04 - accuracy: 0.9997 - val_loss: 0.1126 - val_accuracy: 0.9993 - 249ms/epoch - 2ms/step
Epoch 12/50
154/154 - 0s - loss: 0.0012 - accuracy: 0.9996 - val_loss: 0.1100 - val_accuracy: 0.9993 - 243ms/epoch - 2ms/step
Epoch 13/50
154/154 - 0s - loss: 0.0010 - accuracy: 0.9997 - val_loss: 0.1119 - val_accuracy: 0.9994 - 244ms/epoch - 2ms/step
Epoch 14/50
154/154 - 0s - loss: 6.7628e-04 - accuracy: 0.9998 - val_loss: 0.1102 - val_accuracy: 0.9993 - 249ms/epoch - 2ms/step
Epoch 15/50
154/154 - 0s - loss: 6.4962e-04 - accuracy: 0.9998 - val_loss: 0.0894 - val_accuracy: 0.9994 - 245ms/epoch - 2ms/step
Epoch 16/50
154/154 - 0s - loss: 5.6005e-04 - accuracy: 0.9998 - val_loss: 0.1319 - val_accuracy: 0.9993 - 245ms/epoch - 2ms/step
Epoch 17/50
154/154 - 0s - loss: 5.2412e-04 - accuracy: 0.9998 - val_loss: 0.1003 - val_accuracy: 0.9993 - 245ms/epoch - 2ms/step
Epoch 18/50
154/154 - 0s - loss: 7.8476e-04 - accuracy: 0.9997 - val_loss: 0.1021 - val_accuracy: 0.9994 - 245ms/epoch - 2ms/step
Epoch 19/50
154/154 - 0s - loss: 4.8987e-04 - accuracy: 0.9999 - val_loss: 0.1160 - val_accuracy: 0.9994 - 247ms/epoch - 2ms/step
Epoch 20/50
154/154 - 0s - loss: 4.4970e-04 - accuracy: 0.9999 - val_loss: 0.1272 - val_accuracy: 0.9994 - 247ms/epoch - 2ms/step
Epoch 21/50
154/154 - 0s - loss: 5.1281e-04 - accuracy: 0.9998 - val_loss: 0.1244 - val_accuracy: 0.9994 - 245ms/epoch - 2ms/step
Epoch 22/50
154/154 - 0s - loss: 4.4341e-04 - accuracy: 0.9999 - val_loss: 0.1237 - val_accuracy: 0.9994 - 248ms/epoch - 2ms/step
Epoch 23/50
154/154 - 0s - loss: 5.1135e-04 - accuracy: 0.9998 - val_loss: 0.1358 - val_accuracy: 0.9994 - 247ms/epoch - 2ms/step
Epoch 24/50
154/154 - 0s - loss: 4.1338e-04 - accuracy: 0.9998 - val_loss: 0.1328 - val_accuracy: 0.9994 - 246ms/epoch - 2ms/step
Epoch 25/50
154/154 - 0s - loss: 4.5418e-04 - accuracy: 0.9998 - val_loss: 0.0859 - val_accuracy: 0.9994 - 248ms/epoch - 2ms/step
Epoch 26/50
154/154 - 0s - loss: 3.9497e-04 - accuracy: 0.9999 - val_loss: 0.0984 - val_accuracy: 0.9994 - 248ms/epoch - 2ms/step
Epoch 27/50
154/154 - 0s - loss: 3.3437e-04 - accuracy: 0.9999 - val_loss: 0.1170 - val_accuracy: 0.9993 - 247ms/epoch - 2ms/step
Epoch 28/50
154/154 - 0s - loss: 3.7025e-04 - accuracy: 0.9999 - val_loss: 0.1119 - val_accuracy: 0.9994 - 250ms/epoch - 2ms/step
Epoch 29/50
154/154 - 0s - loss: 3.8669e-04 - accuracy: 0.9999 - val_loss: 0.1003 - val_accuracy: 0.9994 - 254ms/epoch - 2ms/step
Epoch 30/50
154/154 - 0s - loss: 9.7113e-04 - accuracy: 0.9998 - val_loss: 0.1448 - val_accuracy: 0.9994 - 248ms/epoch - 2ms/step
Epoch 31/50
154/154 - 0s - loss: 0.0025 - accuracy: 0.9996 - val_loss: 0.1117 - val_accuracy: 0.9994 - 246ms/epoch - 2ms/step
Epoch 32/50
154/154 - 0s - loss: 8.6676e-04 - accuracy: 0.9998 - val_loss: 0.1130 - val_accuracy: 0.9994 - 246ms/epoch - 2ms/step
Epoch 33/50
154/154 - 0s - loss: 5.6344e-04 - accuracy: 0.9998 - val_loss: 0.1075 - val_accuracy: 0.9994 - 246ms/epoch - 2ms/step
Epoch 34/50
154/154 - 0s - loss: 3.2929e-04 - accuracy: 0.9999 - val_loss: 0.1377 - val_accuracy: 0.9994 - 248ms/epoch - 2ms/step
Epoch 35/50
154/154 - 0s - loss: 3.2917e-04 - accuracy: 0.9999 - val_loss: 0.1338 - val_accuracy: 0.9994 - 249ms/epoch - 2ms/step
Epoch 36/50
154/154 - 0s - loss: 3.8197e-04 - accuracy: 0.9998 - val_loss: 0.1314 - val_accuracy: 0.9994 - 251ms/epoch - 2ms/step
Epoch 37/50
154/154 - 0s - loss: 3.2134e-04 - accuracy: 0.9999 - val_loss: 0.1412 - val_accuracy: 0.9994 - 247ms/epoch - 2ms/step
Epoch 38/50
154/154 - 0s - loss: 2.7892e-04 - accuracy: 0.9999 - val_loss: 0.1432 - val_accuracy: 0.9994 - 247ms/epoch - 2ms/step
Epoch 39/50
154/154 - 0s - loss: 2.8548e-04 - accuracy: 0.9999 - val_loss: 0.1505 - val_accuracy: 0.9994 - 249ms/epoch - 2ms/step
Epoch 40/50
154/154 - 0s - loss: 2.9354e-04 - accuracy: 0.9999 - val_loss: 0.1472 - val_accuracy: 0.9994 - 249ms/epoch - 2ms/step
Epoch 41/50
154/154 - 0s - loss: 3.0325e-04 - accuracy: 0.9999 - val_loss: 0.1764 - val_accuracy: 0.9994 - 249ms/epoch - 2ms/step
Epoch 42/50
154/154 - 0s - loss: 3.5837e-04 - accuracy: 0.9999 - val_loss: 0.1576 - val_accuracy: 0.9994 - 247ms/epoch - 2ms/step
Epoch 43/50
154/154 - 0s - loss: 2.4189e-04 - accuracy: 0.9998 - val_loss: 0.1930 - val_accuracy: 0.9994 - 249ms/epoch - 2ms/step
Epoch 44/50
154/154 - 0s - loss: 2.8302e-04 - accuracy: 0.9999 - val_loss: 0.1760 - val_accuracy: 0.9994 - 246ms/epoch - 2ms/step
Epoch 45/50
154/154 - 0s - loss: 3.0616e-04 - accuracy: 0.9999 - val_loss: 0.1902 - val_accuracy: 0.9994 - 246ms/epoch - 2ms/step
Epoch 46/50
154/154 - 0s - loss: 3.3151e-04 - accuracy: 0.9998 - val_loss: 0.1922 - val_accuracy: 0.9995 - 253ms/epoch - 2ms/step
Epoch 47/50
154/154 - 0s - loss: 6.6426e-04 - accuracy: 0.9998 - val_loss: 0.1897 - val_accuracy: 0.9994 - 251ms/epoch - 2ms/step
Epoch 48/50
154/154 - 0s - loss: 1.9923e-04 - accuracy: 0.9999 - val_loss: 0.2409 - val_accuracy: 0.9995 - 246ms/epoch - 2ms/step
Epoch 49/50
154/154 - 0s - loss: 2.8000e-04 - accuracy: 0.9999 - val_loss: 0.2325 - val_accuracy: 0.9995 - 250ms/epoch - 2ms/step
Epoch 50/50
154/154 - 0s - loss: 2.5175e-04 - accuracy: 0.9999 - val_loss: 0.2554 - val_accuracy: 0.9995 - 246ms/epoch - 2ms/step

Training model with SGD optimizer, he_normal initializer, Sigmoid activation
Epoch 1/50
154/154 - 1s - loss: 0.5015 - accuracy: 0.8018 - val_loss: 0.4963 - val_accuracy: 0.8035 - 723ms/epoch - 5ms/step
Epoch 2/50
154/154 - 0s - loss: 0.4976 - accuracy: 0.8018 - val_loss: 0.4948 - val_accuracy: 0.8035 - 254ms/epoch - 2ms/step
Epoch 3/50
154/154 - 0s - loss: 0.4970 - accuracy: 0.8018 - val_loss: 0.4945 - val_accuracy: 0.8035 - 238ms/epoch - 2ms/step
Epoch 4/50
154/154 - 0s - loss: 0.4967 - accuracy: 0.8018 - val_loss: 0.4942 - val_accuracy: 0.8035 - 237ms/epoch - 2ms/step
Epoch 5/50
154/154 - 0s - loss: 0.4965 - accuracy: 0.8018 - val_loss: 0.4941 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step
Epoch 6/50
154/154 - 0s - loss: 0.4963 - accuracy: 0.8018 - val_loss: 0.4939 - val_accuracy: 0.8035 - 253ms/epoch - 2ms/step
Epoch 7/50
154/154 - 0s - loss: 0.4961 - accuracy: 0.8018 - val_loss: 0.4937 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step
Epoch 8/50
154/154 - 0s - loss: 0.4959 - accuracy: 0.8018 - val_loss: 0.4935 - val_accuracy: 0.8035 - 233ms/epoch - 2ms/step
Epoch 9/50
154/154 - 0s - loss: 0.4957 - accuracy: 0.8018 - val_loss: 0.4933 - val_accuracy: 0.8035 - 257ms/epoch - 2ms/step
Epoch 10/50
154/154 - 0s - loss: 0.4955 - accuracy: 0.8018 - val_loss: 0.4931 - val_accuracy: 0.8035 - 234ms/epoch - 2ms/step
Epoch 11/50
154/154 - 0s - loss: 0.4953 - accuracy: 0.8018 - val_loss: 0.4929 - val_accuracy: 0.8035 - 239ms/epoch - 2ms/step
Epoch 12/50
154/154 - 0s - loss: 0.4951 - accuracy: 0.8018 - val_loss: 0.4927 - val_accuracy: 0.8035 - 237ms/epoch - 2ms/step
Epoch 13/50
154/154 - 0s - loss: 0.4949 - accuracy: 0.8018 - val_loss: 0.4924 - val_accuracy: 0.8035 - 235ms/epoch - 2ms/step
Epoch 14/50
154/154 - 0s - loss: 0.4947 - accuracy: 0.8018 - val_loss: 0.4922 - val_accuracy: 0.8035 - 233ms/epoch - 2ms/step
Epoch 15/50
154/154 - 0s - loss: 0.4944 - accuracy: 0.8018 - val_loss: 0.4920 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step
Epoch 16/50
154/154 - 0s - loss: 0.4942 - accuracy: 0.8018 - val_loss: 0.4917 - val_accuracy: 0.8035 - 236ms/epoch - 2ms/step
Epoch 17/50
154/154 - 0s - loss: 0.4940 - accuracy: 0.8018 - val_loss: 0.4915 - val_accuracy: 0.8035 - 234ms/epoch - 2ms/step
Epoch 18/50
154/154 - 0s - loss: 0.4937 - accuracy: 0.8018 - val_loss: 0.4912 - val_accuracy: 0.8035 - 236ms/epoch - 2ms/step
Epoch 19/50
154/154 - 0s - loss: 0.4934 - accuracy: 0.8018 - val_loss: 0.4910 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step
Epoch 20/50
154/154 - 0s - loss: 0.4931 - accuracy: 0.8018 - val_loss: 0.4907 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step
Epoch 21/50
154/154 - 0s - loss: 0.4928 - accuracy: 0.8018 - val_loss: 0.4904 - val_accuracy: 0.8035 - 231ms/epoch - 1ms/step
Epoch 22/50
154/154 - 0s - loss: 0.4925 - accuracy: 0.8018 - val_loss: 0.4901 - val_accuracy: 0.8035 - 233ms/epoch - 2ms/step
Epoch 23/50
154/154 - 0s - loss: 0.4922 - accuracy: 0.8018 - val_loss: 0.4897 - val_accuracy: 0.8035 - 231ms/epoch - 1ms/step
Epoch 24/50
154/154 - 0s - loss: 0.4919 - accuracy: 0.8018 - val_loss: 0.4894 - val_accuracy: 0.8035 - 224ms/epoch - 1ms/step
Epoch 25/50
154/154 - 0s - loss: 0.4915 - accuracy: 0.8018 - val_loss: 0.4890 - val_accuracy: 0.8035 - 234ms/epoch - 2ms/step
Epoch 26/50
154/154 - 0s - loss: 0.4911 - accuracy: 0.8018 - val_loss: 0.4886 - val_accuracy: 0.8035 - 233ms/epoch - 2ms/step
Epoch 27/50
154/154 - 0s - loss: 0.4908 - accuracy: 0.8018 - val_loss: 0.4882 - val_accuracy: 0.8035 - 230ms/epoch - 1ms/step
Epoch 28/50
154/154 - 0s - loss: 0.4903 - accuracy: 0.8018 - val_loss: 0.4878 - val_accuracy: 0.8035 - 234ms/epoch - 2ms/step
Epoch 29/50
154/154 - 0s - loss: 0.4899 - accuracy: 0.8018 - val_loss: 0.4874 - val_accuracy: 0.8035 - 234ms/epoch - 2ms/step
Epoch 30/50
154/154 - 0s - loss: 0.4894 - accuracy: 0.8018 - val_loss: 0.4869 - val_accuracy: 0.8035 - 234ms/epoch - 2ms/step
Epoch 31/50
154/154 - 0s - loss: 0.4889 - accuracy: 0.8018 - val_loss: 0.4864 - val_accuracy: 0.8035 - 235ms/epoch - 2ms/step
Epoch 32/50
154/154 - 0s - loss: 0.4884 - accuracy: 0.8018 - val_loss: 0.4858 - val_accuracy: 0.8035 - 240ms/epoch - 2ms/step
Epoch 33/50
154/154 - 0s - loss: 0.4878 - accuracy: 0.8018 - val_loss: 0.4852 - val_accuracy: 0.8035 - 231ms/epoch - 2ms/step
Epoch 34/50
154/154 - 0s - loss: 0.4872 - accuracy: 0.8018 - val_loss: 0.4846 - val_accuracy: 0.8035 - 235ms/epoch - 2ms/step
Epoch 35/50
154/154 - 0s - loss: 0.4865 - accuracy: 0.8018 - val_loss: 0.4840 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step
Epoch 36/50
154/154 - 0s - loss: 0.4858 - accuracy: 0.8018 - val_loss: 0.4832 - val_accuracy: 0.8035 - 233ms/epoch - 2ms/step
Epoch 37/50
154/154 - 0s - loss: 0.4851 - accuracy: 0.8018 - val_loss: 0.4825 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step
Epoch 38/50
154/154 - 0s - loss: 0.4843 - accuracy: 0.8018 - val_loss: 0.4816 - val_accuracy: 0.8035 - 235ms/epoch - 2ms/step
Epoch 39/50
154/154 - 0s - loss: 0.4834 - accuracy: 0.8018 - val_loss: 0.4807 - val_accuracy: 0.8035 - 230ms/epoch - 1ms/step
Epoch 40/50
154/154 - 0s - loss: 0.4825 - accuracy: 0.8018 - val_loss: 0.4797 - val_accuracy: 0.8035 - 225ms/epoch - 1ms/step
Epoch 41/50
154/154 - 0s - loss: 0.4814 - accuracy: 0.8018 - val_loss: 0.4787 - val_accuracy: 0.8035 - 237ms/epoch - 2ms/step
Epoch 42/50
154/154 - 0s - loss: 0.4803 - accuracy: 0.8018 - val_loss: 0.4775 - val_accuracy: 0.8035 - 238ms/epoch - 2ms/step
Epoch 43/50
154/154 - 0s - loss: 0.4791 - accuracy: 0.8018 - val_loss: 0.4763 - val_accuracy: 0.8035 - 242ms/epoch - 2ms/step
Epoch 44/50
154/154 - 0s - loss: 0.4777 - accuracy: 0.8018 - val_loss: 0.4749 - val_accuracy: 0.8035 - 238ms/epoch - 2ms/step
Epoch 45/50
154/154 - 0s - loss: 0.4763 - accuracy: 0.8018 - val_loss: 0.4734 - val_accuracy: 0.8035 - 235ms/epoch - 2ms/step
Epoch 46/50
154/154 - 0s - loss: 0.4747 - accuracy: 0.8018 - val_loss: 0.4717 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step
Epoch 47/50
154/154 - 0s - loss: 0.4729 - accuracy: 0.8018 - val_loss: 0.4699 - val_accuracy: 0.8035 - 238ms/epoch - 2ms/step
Epoch 48/50
154/154 - 0s - loss: 0.4710 - accuracy: 0.8018 - val_loss: 0.4678 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step
Epoch 49/50
154/154 - 0s - loss: 0.4688 - accuracy: 0.8018 - val_loss: 0.4656 - val_accuracy: 0.8035 - 233ms/epoch - 2ms/step
Epoch 50/50
154/154 - 0s - loss: 0.4664 - accuracy: 0.8018 - val_loss: 0.4631 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step

Training model with SGD optimizer, glorot_uniform initializer, Sigmoid activation
Epoch 1/50
154/154 - 1s - loss: 0.5501 - accuracy: 0.8018 - val_loss: 0.5091 - val_accuracy: 0.8035 - 674ms/epoch - 4ms/step
Epoch 2/50
154/154 - 0s - loss: 0.5031 - accuracy: 0.8018 - val_loss: 0.4970 - val_accuracy: 0.8035 - 233ms/epoch - 2ms/step
Epoch 3/50
154/154 - 0s - loss: 0.4982 - accuracy: 0.8018 - val_loss: 0.4954 - val_accuracy: 0.8035 - 240ms/epoch - 2ms/step
Epoch 4/50
154/154 - 0s - loss: 0.4976 - accuracy: 0.8018 - val_loss: 0.4951 - val_accuracy: 0.8035 - 234ms/epoch - 2ms/step
Epoch 5/50
154/154 - 0s - loss: 0.4975 - accuracy: 0.8018 - val_loss: 0.4950 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step
Epoch 6/50
154/154 - 0s - loss: 0.4974 - accuracy: 0.8018 - val_loss: 0.4950 - val_accuracy: 0.8035 - 231ms/epoch - 1ms/step
Epoch 7/50
154/154 - 0s - loss: 0.4973 - accuracy: 0.8018 - val_loss: 0.4949 - val_accuracy: 0.8035 - 239ms/epoch - 2ms/step
Epoch 8/50
154/154 - 0s - loss: 0.4973 - accuracy: 0.8018 - val_loss: 0.4948 - val_accuracy: 0.8035 - 234ms/epoch - 2ms/step
Epoch 9/50
154/154 - 0s - loss: 0.4972 - accuracy: 0.8018 - val_loss: 0.4948 - val_accuracy: 0.8035 - 237ms/epoch - 2ms/step
Epoch 10/50
154/154 - 0s - loss: 0.4972 - accuracy: 0.8018 - val_loss: 0.4947 - val_accuracy: 0.8035 - 237ms/epoch - 2ms/step
Epoch 11/50
154/154 - 0s - loss: 0.4971 - accuracy: 0.8018 - val_loss: 0.4946 - val_accuracy: 0.8035 - 230ms/epoch - 1ms/step
Epoch 12/50
154/154 - 0s - loss: 0.4970 - accuracy: 0.8018 - val_loss: 0.4946 - val_accuracy: 0.8035 - 226ms/epoch - 1ms/step
Epoch 13/50
154/154 - 0s - loss: 0.4970 - accuracy: 0.8018 - val_loss: 0.4945 - val_accuracy: 0.8035 - 228ms/epoch - 1ms/step
Epoch 14/50
154/154 - 0s - loss: 0.4969 - accuracy: 0.8018 - val_loss: 0.4945 - val_accuracy: 0.8035 - 235ms/epoch - 2ms/step
Epoch 15/50
154/154 - 0s - loss: 0.4969 - accuracy: 0.8018 - val_loss: 0.4944 - val_accuracy: 0.8035 - 229ms/epoch - 1ms/step
Epoch 16/50
154/154 - 0s - loss: 0.4968 - accuracy: 0.8018 - val_loss: 0.4944 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step
Epoch 17/50
154/154 - 0s - loss: 0.4967 - accuracy: 0.8018 - val_loss: 0.4943 - val_accuracy: 0.8035 - 229ms/epoch - 1ms/step
Epoch 18/50
154/154 - 0s - loss: 0.4967 - accuracy: 0.8018 - val_loss: 0.4942 - val_accuracy: 0.8035 - 233ms/epoch - 2ms/step
Epoch 19/50
154/154 - 0s - loss: 0.4966 - accuracy: 0.8018 - val_loss: 0.4942 - val_accuracy: 0.8035 - 230ms/epoch - 1ms/step
Epoch 20/50
154/154 - 0s - loss: 0.4965 - accuracy: 0.8018 - val_loss: 0.4941 - val_accuracy: 0.8035 - 233ms/epoch - 2ms/step
Epoch 21/50
154/154 - 0s - loss: 0.4965 - accuracy: 0.8018 - val_loss: 0.4940 - val_accuracy: 0.8035 - 230ms/epoch - 1ms/step
Epoch 22/50
154/154 - 0s - loss: 0.4964 - accuracy: 0.8018 - val_loss: 0.4939 - val_accuracy: 0.8035 - 231ms/epoch - 2ms/step
Epoch 23/50
154/154 - 0s - loss: 0.4963 - accuracy: 0.8018 - val_loss: 0.4939 - val_accuracy: 0.8035 - 234ms/epoch - 2ms/step
Epoch 24/50
154/154 - 0s - loss: 0.4962 - accuracy: 0.8018 - val_loss: 0.4938 - val_accuracy: 0.8035 - 233ms/epoch - 2ms/step
Epoch 25/50
154/154 - 0s - loss: 0.4962 - accuracy: 0.8018 - val_loss: 0.4937 - val_accuracy: 0.8035 - 248ms/epoch - 2ms/step
Epoch 26/50
154/154 - 0s - loss: 0.4961 - accuracy: 0.8018 - val_loss: 0.4936 - val_accuracy: 0.8035 - 233ms/epoch - 2ms/step
Epoch 27/50
154/154 - 0s - loss: 0.4960 - accuracy: 0.8018 - val_loss: 0.4935 - val_accuracy: 0.8035 - 229ms/epoch - 1ms/step
Epoch 28/50
154/154 - 0s - loss: 0.4959 - accuracy: 0.8018 - val_loss: 0.4935 - val_accuracy: 0.8035 - 230ms/epoch - 1ms/step
Epoch 29/50
154/154 - 0s - loss: 0.4958 - accuracy: 0.8018 - val_loss: 0.4934 - val_accuracy: 0.8035 - 229ms/epoch - 1ms/step
Epoch 30/50
154/154 - 0s - loss: 0.4957 - accuracy: 0.8018 - val_loss: 0.4933 - val_accuracy: 0.8035 - 232ms/epoch - 2ms/step
Epoch 31/50
154/154 - 0s - loss: 0.4957 - accuracy: 0.8018 - val_loss: 0.4932 - val_accuracy: 0.8035 - 231ms/epoch - 1ms/step
Epoch 32/50
154/154 - 0s - loss: 0.4956 - accuracy: 0.8018 - val_loss: 0.4931 - val_accuracy: 0.8035 - 238ms/epoch - 2ms/step
Epoch 33/50
154/154 - 0s - loss: 0.4955 - accuracy: 0.8018 - val_loss: 0.4930 - val_accuracy: 0.8035 - 235ms/epoch - 2ms/step
Epoch 34/50
154/154 - 0s - loss: 0.4954 - accuracy: 0.8018 - val_loss: 0.4929 - val_accuracy: 0.8035 - 236ms/epoch - 2ms/step
Epoch 35/50
154/154 - 0s - loss: 0.4953 - accuracy: 0.8018 - val_loss: 0.4928 - val_accuracy: 0.8035 - 235ms/epoch - 2ms/step
Epoch 36/50
154/154 - 0s - loss: 0.4952 - accuracy: 0.8018 - val_loss: 0.4927 - val_accuracy: 0.8035 - 237ms/epoch - 2ms/step
Epoch 37/50
154/154 - 0s - loss: 0.4951 - accuracy: 0.8018 - val_loss: 0.4926 - val_accuracy: 0.8035 - 235ms/epoch - 2ms/step
Epoch 38/50
154/154 - 0s - loss: 0.4949 - accuracy: 0.8018 - val_loss: 0.4925 - val_accuracy: 0.8035 - 231ms/epoch - 1ms/step
Epoch 39/50
154/154 - 0s - loss: 0.4948 - accuracy: 0.8018 - val_loss: 0.4924 - val_accuracy: 0.8035 - 239ms/epoch - 2ms/step
Epoch 40/50
154/154 - 0s - loss: 0.4947 - accuracy: 0.8018 - val_loss: 0.4923 - val_accuracy: 0.8035 - 230ms/epoch - 1ms/step
Epoch 41/50
154/154 - 0s - loss: 0.4946 - accuracy: 0.8018 - val_loss: 0.4921 - val_accuracy: 0.8035 - 228ms/epoch - 1ms/step
Epoch 42/50
154/154 - 0s - loss: 0.4944 - accuracy: 0.8018 - val_loss: 0.4920 - val_accuracy: 0.8035 - 234ms/epoch - 2ms/step
Epoch 43/50
154/154 - 0s - loss: 0.4943 - accuracy: 0.8018 - val_loss: 0.4918 - val_accuracy: 0.8035 - 235ms/epoch - 2ms/step
Epoch 44/50
154/154 - 0s - loss: 0.4942 - accuracy: 0.8018 - val_loss: 0.4917 - val_accuracy: 0.8035 - 230ms/epoch - 1ms/step
Epoch 45/50
154/154 - 0s - loss: 0.4940 - accuracy: 0.8018 - val_loss: 0.4916 - val_accuracy: 0.8035 - 230ms/epoch - 1ms/step
Epoch 46/50
154/154 - 0s - loss: 0.4939 - accuracy: 0.8018 - val_loss: 0.4914 - val_accuracy: 0.8035 - 230ms/epoch - 1ms/step
Epoch 47/50
154/154 - 0s - loss: 0.4937 - accuracy: 0.8018 - val_loss: 0.4912 - val_accuracy: 0.8035 - 229ms/epoch - 1ms/step
Epoch 48/50
154/154 - 0s - loss: 0.4935 - accuracy: 0.8018 - val_loss: 0.4911 - val_accuracy: 0.8035 - 234ms/epoch - 2ms/step
Epoch 49/50
154/154 - 0s - loss: 0.4934 - accuracy: 0.8018 - val_loss: 0.4909 - val_accuracy: 0.8035 - 233ms/epoch - 2ms/step
Epoch 50/50
154/154 - 0s - loss: 0.4932 - accuracy: 0.8018 - val_loss: 0.4907 - val_accuracy: 0.8035 - 234ms/epoch - 2ms/step

Training model with Adam optimizer, he_normal initializer, Sigmoid activation
Epoch 1/50
154/154 - 1s - loss: 0.2537 - accuracy: 0.9147 - val_loss: 0.0756 - val_accuracy: 0.9965 - 765ms/epoch - 5ms/step
Epoch 2/50
154/154 - 0s - loss: 0.0551 - accuracy: 0.9967 - val_loss: 0.0430 - val_accuracy: 0.9967 - 252ms/epoch - 2ms/step
Epoch 3/50
154/154 - 0s - loss: 0.0369 - accuracy: 0.9968 - val_loss: 0.0325 - val_accuracy: 0.9968 - 259ms/epoch - 2ms/step
Epoch 4/50
154/154 - 0s - loss: 0.0294 - accuracy: 0.9970 - val_loss: 0.0273 - val_accuracy: 0.9969 - 248ms/epoch - 2ms/step
Epoch 5/50
154/154 - 0s - loss: 0.0253 - accuracy: 0.9970 - val_loss: 0.0244 - val_accuracy: 0.9969 - 250ms/epoch - 2ms/step
Epoch 6/50
154/154 - 0s - loss: 0.0224 - accuracy: 0.9970 - val_loss: 0.0206 - val_accuracy: 0.9969 - 267ms/epoch - 2ms/step
Epoch 7/50
154/154 - 0s - loss: 0.0140 - accuracy: 0.9991 - val_loss: 0.0134 - val_accuracy: 0.9993 - 247ms/epoch - 2ms/step
Epoch 8/50
154/154 - 0s - loss: 0.0110 - accuracy: 0.9996 - val_loss: 0.0119 - val_accuracy: 0.9992 - 249ms/epoch - 2ms/step
Epoch 9/50
154/154 - 0s - loss: 0.0094 - accuracy: 0.9996 - val_loss: 0.0108 - val_accuracy: 0.9991 - 252ms/epoch - 2ms/step
Epoch 10/50
154/154 - 0s - loss: 0.0085 - accuracy: 0.9995 - val_loss: 0.0097 - val_accuracy: 0.9993 - 247ms/epoch - 2ms/step
Epoch 11/50
154/154 - 0s - loss: 0.0074 - accuracy: 0.9996 - val_loss: 0.0090 - val_accuracy: 0.9993 - 246ms/epoch - 2ms/step
Epoch 12/50
154/154 - 0s - loss: 0.0065 - accuracy: 0.9996 - val_loss: 0.0083 - val_accuracy: 0.9993 - 261ms/epoch - 2ms/step
Epoch 13/50
154/154 - 0s - loss: 0.0059 - accuracy: 0.9996 - val_loss: 0.0077 - val_accuracy: 0.9993 - 259ms/epoch - 2ms/step
Epoch 14/50
154/154 - 0s - loss: 0.0054 - accuracy: 0.9996 - val_loss: 0.0075 - val_accuracy: 0.9993 - 254ms/epoch - 2ms/step
Epoch 15/50
154/154 - 0s - loss: 0.0050 - accuracy: 0.9996 - val_loss: 0.0069 - val_accuracy: 0.9994 - 258ms/epoch - 2ms/step
Epoch 16/50
154/154 - 0s - loss: 0.0046 - accuracy: 0.9997 - val_loss: 0.0066 - val_accuracy: 0.9994 - 260ms/epoch - 2ms/step
Epoch 17/50
154/154 - 0s - loss: 0.0042 - accuracy: 0.9997 - val_loss: 0.0065 - val_accuracy: 0.9993 - 248ms/epoch - 2ms/step
Epoch 18/50
154/154 - 0s - loss: 0.0040 - accuracy: 0.9997 - val_loss: 0.0062 - val_accuracy: 0.9993 - 248ms/epoch - 2ms/step
Epoch 19/50
154/154 - 0s - loss: 0.0038 - accuracy: 0.9996 - val_loss: 0.0059 - val_accuracy: 0.9994 - 250ms/epoch - 2ms/step
Epoch 20/50
154/154 - 0s - loss: 0.0034 - accuracy: 0.9998 - val_loss: 0.0058 - val_accuracy: 0.9993 - 251ms/epoch - 2ms/step
Epoch 21/50
154/154 - 0s - loss: 0.0033 - accuracy: 0.9997 - val_loss: 0.0057 - val_accuracy: 0.9993 - 251ms/epoch - 2ms/step
Epoch 22/50
154/154 - 0s - loss: 0.0031 - accuracy: 0.9997 - val_loss: 0.0055 - val_accuracy: 0.9994 - 250ms/epoch - 2ms/step
Epoch 23/50
154/154 - 0s - loss: 0.0029 - accuracy: 0.9998 - val_loss: 0.0054 - val_accuracy: 0.9994 - 250ms/epoch - 2ms/step
Epoch 24/50
154/154 - 0s - loss: 0.0028 - accuracy: 0.9997 - val_loss: 0.0054 - val_accuracy: 0.9994 - 251ms/epoch - 2ms/step
Epoch 25/50
154/154 - 0s - loss: 0.0027 - accuracy: 0.9997 - val_loss: 0.0053 - val_accuracy: 0.9993 - 255ms/epoch - 2ms/step
Epoch 26/50
154/154 - 0s - loss: 0.0026 - accuracy: 0.9997 - val_loss: 0.0052 - val_accuracy: 0.9994 - 254ms/epoch - 2ms/step
Epoch 27/50
154/154 - 0s - loss: 0.0024 - accuracy: 0.9997 - val_loss: 0.0052 - val_accuracy: 0.9994 - 250ms/epoch - 2ms/step
Epoch 28/50
154/154 - 0s - loss: 0.0022 - accuracy: 0.9997 - val_loss: 0.0051 - val_accuracy: 0.9994 - 254ms/epoch - 2ms/step
Epoch 29/50
154/154 - 0s - loss: 0.0020 - accuracy: 0.9998 - val_loss: 0.0052 - val_accuracy: 0.9993 - 249ms/epoch - 2ms/step
Epoch 30/50
154/154 - 0s - loss: 0.0020 - accuracy: 0.9998 - val_loss: 0.0051 - val_accuracy: 0.9994 - 250ms/epoch - 2ms/step
Epoch 31/50
154/154 - 0s - loss: 0.0022 - accuracy: 0.9997 - val_loss: 0.0051 - val_accuracy: 0.9994 - 252ms/epoch - 2ms/step
Epoch 32/50
154/154 - 0s - loss: 0.0018 - accuracy: 0.9998 - val_loss: 0.0050 - val_accuracy: 0.9994 - 250ms/epoch - 2ms/step
Epoch 33/50
154/154 - 0s - loss: 0.0018 - accuracy: 0.9998 - val_loss: 0.0050 - val_accuracy: 0.9994 - 247ms/epoch - 2ms/step
Epoch 34/50
154/154 - 0s - loss: 0.0017 - accuracy: 0.9998 - val_loss: 0.0050 - val_accuracy: 0.9994 - 251ms/epoch - 2ms/step
Epoch 35/50
154/154 - 0s - loss: 0.0017 - accuracy: 0.9998 - val_loss: 0.0050 - val_accuracy: 0.9994 - 250ms/epoch - 2ms/step
Epoch 36/50
154/154 - 0s - loss: 0.0016 - accuracy: 0.9998 - val_loss: 0.0050 - val_accuracy: 0.9994 - 250ms/epoch - 2ms/step
Epoch 37/50
154/154 - 0s - loss: 0.0016 - accuracy: 0.9998 - val_loss: 0.0050 - val_accuracy: 0.9994 - 249ms/epoch - 2ms/step
Epoch 38/50
154/154 - 0s - loss: 0.0016 - accuracy: 0.9998 - val_loss: 0.0050 - val_accuracy: 0.9994 - 253ms/epoch - 2ms/step
Epoch 39/50
154/154 - 0s - loss: 0.0015 - accuracy: 0.9998 - val_loss: 0.0050 - val_accuracy: 0.9994 - 250ms/epoch - 2ms/step
Epoch 40/50
154/154 - 0s - loss: 0.0016 - accuracy: 0.9998 - val_loss: 0.0050 - val_accuracy: 0.9994 - 243ms/epoch - 2ms/step
Epoch 41/50
154/154 - 0s - loss: 0.0016 - accuracy: 0.9998 - val_loss: 0.0050 - val_accuracy: 0.9994 - 244ms/epoch - 2ms/step
Epoch 42/50
154/154 - 0s - loss: 0.0015 - accuracy: 0.9998 - val_loss: 0.0050 - val_accuracy: 0.9994 - 241ms/epoch - 2ms/step
Epoch 43/50
154/154 - 44s - loss: 0.0014 - accuracy: 0.9998 - val_loss: 0.0050 - val_accuracy: 0.9994 - 44s/epoch - 287ms/step
Epoch 44/50
154/154 - 0s - loss: 0.0014 - accuracy: 0.9998 - val_loss: 0.0051 - val_accuracy: 0.9994 - 413ms/epoch - 3ms/step
Epoch 45/50
154/154 - 1s - loss: 0.0014 - accuracy: 0.9998 - val_loss: 0.0051 - val_accuracy: 0.9994 - 748ms/epoch - 5ms/step
Epoch 46/50
154/154 - 0s - loss: 0.0013 - accuracy: 0.9998 - val_loss: 0.0054 - val_accuracy: 0.9993 - 344ms/epoch - 2ms/step
Epoch 47/50
154/154 - 0s - loss: 0.0015 - accuracy: 0.9998 - val_loss: 0.0051 - val_accuracy: 0.9994 - 341ms/epoch - 2ms/step
Epoch 48/50
154/154 - 0s - loss: 0.0014 - accuracy: 0.9998 - val_loss: 0.0053 - val_accuracy: 0.9993 - 291ms/epoch - 2ms/step
Epoch 49/50
154/154 - 0s - loss: 0.0015 - accuracy: 0.9997 - val_loss: 0.0052 - val_accuracy: 0.9993 - 279ms/epoch - 2ms/step
Epoch 50/50
154/154 - 0s - loss: 0.0014 - accuracy: 0.9998 - val_loss: 0.0051 - val_accuracy: 0.9994 - 278ms/epoch - 2ms/step

Training model with Adam optimizer, glorot_uniform initializer, Sigmoid activation
Epoch 1/50
154/154 - 1s - loss: 0.5492 - accuracy: 0.7402 - val_loss: 0.4134 - val_accuracy: 0.8035 - 1s/epoch - 7ms/step
Epoch 2/50
154/154 - 0s - loss: 0.2474 - accuracy: 0.9554 - val_loss: 0.1419 - val_accuracy: 0.9980 - 275ms/epoch - 2ms/step
Epoch 3/50
154/154 - 0s - loss: 0.1018 - accuracy: 0.9988 - val_loss: 0.0750 - val_accuracy: 0.9991 - 274ms/epoch - 2ms/step
Epoch 4/50
154/154 - 0s - loss: 0.0608 - accuracy: 0.9993 - val_loss: 0.0509 - val_accuracy: 0.9990 - 289ms/epoch - 2ms/step
Epoch 5/50
154/154 - 0s - loss: 0.0433 - accuracy: 0.9994 - val_loss: 0.0386 - val_accuracy: 0.9991 - 278ms/epoch - 2ms/step
Epoch 6/50
154/154 - 0s - loss: 0.0336 - accuracy: 0.9994 - val_loss: 0.0310 - val_accuracy: 0.9991 - 463ms/epoch - 3ms/step
Epoch 7/50
154/154 - 0s - loss: 0.0275 - accuracy: 0.9993 - val_loss: 0.0259 - val_accuracy: 0.9991 - 366ms/epoch - 2ms/step
Epoch 8/50
154/154 - 0s - loss: 0.0230 - accuracy: 0.9994 - val_loss: 0.0223 - val_accuracy: 0.9991 - 278ms/epoch - 2ms/step
Epoch 9/50
154/154 - 0s - loss: 0.0198 - accuracy: 0.9994 - val_loss: 0.0195 - val_accuracy: 0.9992 - 305ms/epoch - 2ms/step
Epoch 10/50
154/154 - 0s - loss: 0.0173 - accuracy: 0.9994 - val_loss: 0.0173 - val_accuracy: 0.9992 - 319ms/epoch - 2ms/step
Epoch 11/50
154/154 - 0s - loss: 0.0153 - accuracy: 0.9994 - val_loss: 0.0156 - val_accuracy: 0.9991 - 264ms/epoch - 2ms/step
Epoch 12/50
154/154 - 0s - loss: 0.0137 - accuracy: 0.9994 - val_loss: 0.0144 - val_accuracy: 0.9991 - 269ms/epoch - 2ms/step
Epoch 13/50
154/154 - 0s - loss: 0.0124 - accuracy: 0.9994 - val_loss: 0.0127 - val_accuracy: 0.9992 - 271ms/epoch - 2ms/step
Epoch 14/50
154/154 - 0s - loss: 0.0110 - accuracy: 0.9995 - val_loss: 0.0115 - val_accuracy: 0.9993 - 280ms/epoch - 2ms/step
Epoch 15/50
154/154 - 0s - loss: 0.0100 - accuracy: 0.9994 - val_loss: 0.0106 - val_accuracy: 0.9994 - 265ms/epoch - 2ms/step
Epoch 16/50
154/154 - 0s - loss: 0.0090 - accuracy: 0.9995 - val_loss: 0.0099 - val_accuracy: 0.9994 - 276ms/epoch - 2ms/step
Epoch 17/50
154/154 - 0s - loss: 0.0082 - accuracy: 0.9996 - val_loss: 0.0092 - val_accuracy: 0.9994 - 277ms/epoch - 2ms/step
Epoch 18/50
154/154 - 0s - loss: 0.0075 - accuracy: 0.9996 - val_loss: 0.0087 - val_accuracy: 0.9994 - 268ms/epoch - 2ms/step
Epoch 19/50
154/154 - 0s - loss: 0.0069 - accuracy: 0.9997 - val_loss: 0.0083 - val_accuracy: 0.9993 - 265ms/epoch - 2ms/step
Epoch 20/50
154/154 - 0s - loss: 0.0065 - accuracy: 0.9996 - val_loss: 0.0081 - val_accuracy: 0.9993 - 281ms/epoch - 2ms/step
Epoch 21/50
154/154 - 0s - loss: 0.0062 - accuracy: 0.9995 - val_loss: 0.0075 - val_accuracy: 0.9993 - 266ms/epoch - 2ms/step
Epoch 22/50
154/154 - 0s - loss: 0.0056 - accuracy: 0.9996 - val_loss: 0.0073 - val_accuracy: 0.9993 - 271ms/epoch - 2ms/step
Epoch 23/50
154/154 - 0s - loss: 0.0052 - accuracy: 0.9997 - val_loss: 0.0070 - val_accuracy: 0.9993 - 266ms/epoch - 2ms/step
Epoch 24/50
154/154 - 0s - loss: 0.0049 - accuracy: 0.9996 - val_loss: 0.0066 - val_accuracy: 0.9994 - 261ms/epoch - 2ms/step
Epoch 25/50
154/154 - 0s - loss: 0.0049 - accuracy: 0.9996 - val_loss: 0.0069 - val_accuracy: 0.9992 - 266ms/epoch - 2ms/step
Epoch 26/50
154/154 - 0s - loss: 0.0044 - accuracy: 0.9997 - val_loss: 0.0063 - val_accuracy: 0.9994 - 267ms/epoch - 2ms/step
Epoch 27/50
154/154 - 0s - loss: 0.0042 - accuracy: 0.9997 - val_loss: 0.0061 - val_accuracy: 0.9994 - 267ms/epoch - 2ms/step
Epoch 28/50
154/154 - 0s - loss: 0.0040 - accuracy: 0.9997 - val_loss: 0.0060 - val_accuracy: 0.9994 - 271ms/epoch - 2ms/step
Epoch 29/50
154/154 - 0s - loss: 0.0039 - accuracy: 0.9997 - val_loss: 0.0058 - val_accuracy: 0.9994 - 268ms/epoch - 2ms/step
Epoch 30/50
154/154 - 0s - loss: 0.0037 - accuracy: 0.9997 - val_loss: 0.0057 - val_accuracy: 0.9994 - 267ms/epoch - 2ms/step
Epoch 31/50
154/154 - 0s - loss: 0.0036 - accuracy: 0.9997 - val_loss: 0.0056 - val_accuracy: 0.9994 - 272ms/epoch - 2ms/step
Epoch 32/50
154/154 - 0s - loss: 0.0035 - accuracy: 0.9997 - val_loss: 0.0055 - val_accuracy: 0.9994 - 262ms/epoch - 2ms/step
Epoch 33/50
154/154 - 0s - loss: 0.0033 - accuracy: 0.9997 - val_loss: 0.0055 - val_accuracy: 0.9993 - 266ms/epoch - 2ms/step
Epoch 34/50
154/154 - 0s - loss: 0.0032 - accuracy: 0.9997 - val_loss: 0.0054 - val_accuracy: 0.9994 - 269ms/epoch - 2ms/step
Epoch 35/50
154/154 - 0s - loss: 0.0031 - accuracy: 0.9997 - val_loss: 0.0053 - val_accuracy: 0.9994 - 271ms/epoch - 2ms/step
Epoch 36/50
154/154 - 0s - loss: 0.0031 - accuracy: 0.9997 - val_loss: 0.0054 - val_accuracy: 0.9993 - 269ms/epoch - 2ms/step
Epoch 37/50
154/154 - 0s - loss: 0.0030 - accuracy: 0.9996 - val_loss: 0.0055 - val_accuracy: 0.9992 - 266ms/epoch - 2ms/step
Epoch 38/50
154/154 - 0s - loss: 0.0029 - accuracy: 0.9997 - val_loss: 0.0055 - val_accuracy: 0.9992 - 265ms/epoch - 2ms/step
Epoch 39/50
154/154 - 0s - loss: 0.0029 - accuracy: 0.9996 - val_loss: 0.0054 - val_accuracy: 0.9993 - 267ms/epoch - 2ms/step
Epoch 40/50
154/154 - 0s - loss: 0.0029 - accuracy: 0.9996 - val_loss: 0.0051 - val_accuracy: 0.9994 - 265ms/epoch - 2ms/step
Epoch 41/50
154/154 - 0s - loss: 0.0026 - accuracy: 0.9997 - val_loss: 0.0054 - val_accuracy: 0.9992 - 261ms/epoch - 2ms/step
Epoch 42/50
154/154 - 0s - loss: 0.0026 - accuracy: 0.9997 - val_loss: 0.0073 - val_accuracy: 0.9990 - 273ms/epoch - 2ms/step
Epoch 43/50
154/154 - 0s - loss: 0.0024 - accuracy: 0.9997 - val_loss: 0.0051 - val_accuracy: 0.9993 - 274ms/epoch - 2ms/step
Epoch 44/50
154/154 - 0s - loss: 0.0022 - accuracy: 0.9997 - val_loss: 0.0052 - val_accuracy: 0.9993 - 272ms/epoch - 2ms/step
Epoch 45/50
154/154 - 0s - loss: 0.0022 - accuracy: 0.9997 - val_loss: 0.0052 - val_accuracy: 0.9993 - 264ms/epoch - 2ms/step
Epoch 46/50
154/154 - 0s - loss: 0.0021 - accuracy: 0.9997 - val_loss: 0.0050 - val_accuracy: 0.9994 - 310ms/epoch - 2ms/step
Epoch 47/50
154/154 - 0s - loss: 0.0021 - accuracy: 0.9997 - val_loss: 0.0051 - val_accuracy: 0.9994 - 289ms/epoch - 2ms/step
Epoch 48/50
154/154 - 0s - loss: 0.0021 - accuracy: 0.9997 - val_loss: 0.0052 - val_accuracy: 0.9994 - 272ms/epoch - 2ms/step
Epoch 49/50
154/154 - 0s - loss: 0.0022 - accuracy: 0.9997 - val_loss: 0.0051 - val_accuracy: 0.9993 - 282ms/epoch - 2ms/step
Epoch 50/50
154/154 - 0s - loss: 0.0020 - accuracy: 0.9997 - val_loss: 0.0051 - val_accuracy: 0.9994 - 294ms/epoch - 2ms/step
```
#### 模型描述

该模型是一个全连接的神经网络，其目的是对 KDD Cup 1999 数据集中的攻击类型进行分类（攻击 vs. 正常）。模型使用 TensorFlow 进行实现，采用了 **ReLU** 或 **Sigmoid** 作为激活函数，支持不同的初始化方法和优化器。

#### 层数与每一层的参数

该模型包含 5 层：
- **输入层**（Layer 1）
- **隐藏层**（Layer 2, 3, 4）
- **输出层**（Layer 5）

具体每一层的参数如下：

- **输入层**（Layer 1）：
  - **输入维度**：由数据集特征的数量决定，为42维。
  - **输出维度**：36（每层的节点数）。

- **隐藏层 2**（Layer 2）：
  - **输入维度**：上一层的输出维度（36）。
  - **输出维度**：24。

- **隐藏层 3**（Layer 3）：
  - **输入维度**：上一层的输出维度（24）。
  - **输出维度**：12。

- **隐藏层 4**（Layer 4）：
  - **输入维度**：上一层的输出维度（12）。
  - **输出维度**：6。

- **输出层**（Layer 5）：
  - **输入维度**：上一层的输出维度（6）。
  - **输出维度**：1（因为是二分类问题，输出 0 或 1）。

#### 激活函数选择

- **ReLU** (`relu`)：在前向传播过程中，`ReLU` 激活函数会将负值变为 0，从而帮助神经网络避免梯度消失问题，尤其适用于深层网络。
  
- **Sigmoid** (`sigmoid`)：通过压缩输出到 0 到 1 之间，适合用于二分类问题。输出的值可以被视为预测某一类的概率。

#### 损失函数设置

损失函数采用二分类交叉熵损失函数，它适用于二分类问题。其公式如下：

$$
\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

- 其中，$ y_i $ 是真实标签，$ \hat{y}_i $ 是模型的预测概率（通过 Sigmoid 激活得到的输出）。
- 该损失函数通过最小化预测值与真实标签之间的差异，来优化模型参数。

#### 优化器选择

- **SGD** (`tf.keras.optimizers.SGD`)：随机梯度下降法，通过迭代优化损失函数，适合较小或简洁的模型。
  
- **Adam** (`tf.keras.optimizers.Adam`)：自适应矩估计优化器，结合了动量和自适应学习率，适用于大多数深度学习任务，通常能够更快速地收敛。

#### 参数初始化方法

- **He Normal 初始化**
He Normal 初始化是一种根据层的输入节点数来设置权重初始值的方法。具体来说，权重值从一个以 0 为均值、标准差为 $ \sqrt{2/n_{\text{in}}} $ 的正态分布中随机生成，其中 $ n_{\text{in}} $ 是该层输入节点的数量。
它通常用于带有 ReLU 激活函数的神经网络。它有助于避免梯度消失或爆炸问题，使得网络能够更有效地学习深层网络的参数。
公式为：

$$
W \sim \mathcal{N}(0, \frac{2}{n_{\text{in}}})
$$

其中 $ W $ 是权重，$ n_{\text{in}} $ 是输入节点的数量。

- **Glorot Uniform 初始化**
Glorot Uniform 初始化（也称为 Xavier 初始化）是一种根据层的输入和输出节点数来设置权重初始值的方法。权重值从一个均匀分布中随机生成，该分布的范围是 $ [- \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}] $，其中 $ n_{\text{in}} $ 是输入节点的数量，$ n_{\text{out}} $ 是输出节点的数量。通常用于带有 Sigmoid 或 Tanh 激活函数的神经网络。它有助于保持网络中各层的梯度稳定，避免梯度过大或过小的问题。
公式为：

$$
W \sim U\left(- \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)
$$

其中 $ W $ 是权重，$ n_{\text{in}} $ 和 $ n_{\text{out}} $ 分别是输入和输出节点的数量。
#### 可视化
针对不同⽅法组合（⾄少 4 个组合），plot 出随着 epoch 增⻓ training error 和 test error 的变化情况如下：
![替代文本](loss_Adam_glorot_uniform_ReLU.png)
![替代文本](loss_Adam_glorot_uniform_Sigmoid.png)
![替代文本](loss_Adam_he_normal_ReLU.png)
![替代文本](loss_Adam_he_normal_Sigmoid.png)
![替代文本](loss_SGD_glorot_uniform_ReLU.png)
![替代文本](loss_SGD_glorot_uniform_Sigmoid.png)
![替代文本](loss_SGD_he_normal_ReLU.png)
![替代文本](loss_SGD_he_normal_Sigmoid.png)
