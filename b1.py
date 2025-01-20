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