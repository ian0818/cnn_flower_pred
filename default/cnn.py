# 引入函式庫
# TensorFlow and tf.keras
import tensorflow as tf
import pathlib
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense


# 當前系統資料集絕對路徑
path = 'd:/tf/CNNmodel/newpic'
# 當前系統路徑
load_path = 'd:/tf/CNNmodel/'

#資料參數
size = 32
batch_size = 3772
img_width = 320
img_height = 240

# 檔案系統路徑
print(pathlib.Path(__file__).parent.absolute())
data_dir = pathlib.Path(path)

# 從資料集中取出訓練資料
# 使用 image_dataset_from_directory 函數載入圖片(圖片目錄,驗證數據資料,標籤名稱,改組和轉換的可選隨機種子,從磁盤讀取圖像後調整圖像大小的大小,每批輸入的數量)
# seed 參數，在載入資料集時，將使用指定的種子初始化隨機數生成器，可以保證每次載入資料集的順序都是一致的。
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_width,img_height ),
    batch_size=size
)

# 測試資料
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_width,img_height ),
    batch_size=size
)

#  五項花的檔案名稱
class_names = train_ds.class_names
image_batch, label_batch = next(iter(train_ds))
#  輸出神經元數量
output_number = len(class_names)
# print(output_number)
 # 建立Keras的Sequential模型 Sequential是一個模塊的容器，可以同時運行
 #CNN
model = tf.keras.Sequential([
    # 特徵標準化，可提升模型預測的準確度，梯度運算時也能更快收斂
    # 對輸入的張量進行縮放。將輸入的張量除以 255，這將會將輸入的值縮放到 0 到 1 之間。
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    # max pooling是降躁跟減少運算資源
    # 一個 Convolution Operation 搭配 一個 Pooling
    #卷積
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #池化
    tf.keras.layers.MaxPooling2D(),
    #卷積
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #池化
    tf.keras.layers.MaxPooling2D(),
    #卷積
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #池化
    tf.keras.layers.MaxPooling2D(),
    # 攤平
    tf.keras.layers.Flatten(),
    # 全連接層
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(output_number)
])

# 模型編譯
# 評估模型的評估方式會以accuracy作為標準
# loss from_logits通知損失函數模型生成的輸出值未歸一化，還沒有對它們應用 softmax 函數來產生概率分佈
model.compile(
    #loss:損失函數選擇, optimizer:優化器選擇, metrics:評估標準選擇
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
# 查看模型架構

# 訓練模型
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
)
# print(model.shape)
# model.summary()
model.save(load_path+'model.h5')

# 加載模型
rank_model = keras.models.load_model(load_path+'/model.h5')
# 讓模型輸出的機率值分布在0-1之間
probability_rank_model = tf.keras.Sequential([rank_model, tf.keras.layers.Softmax()])
 
class_names = ['daisy_new','dandelion_new','rose_new','sunflower_new','tulip_new']
for i in range(10):
    # 0到5隨機檔案取樣
    range_num = random.sample(range(5), 1)

    # 隨機取樣之檔案路徑
    file_path = path + '/' +class_names[range_num[0]]


    # 計算圖片檔案數量
    train_Bacterialblight = 0
    for Bacterialblight in os.listdir(file_path):
        train_Bacterialblight = train_Bacterialblight+1

    # 隨機取一張圖片
    img_random = random.sample(range(train_Bacterialblight), 1)
    img_file = file_path+ '/' + str(img_random[0]) + '.jpg'
    print(img_file)

    # Load圖片
    img = keras.preprocessing.image.load_img(img_file, target_size=(320, 240))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # 預測
    pred = probability_rank_model.predict(x)[0]

    top_inds = pred.argsort()[::-1][:5]
    if pred[top_inds[0]] > 0.8:
        print('result is: ' + class_names[top_inds[0]])
    # 印出機率分布
    for data in top_inds:
        print('    {:.3f}  {}'.format(pred[data], class_names[data]))