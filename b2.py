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