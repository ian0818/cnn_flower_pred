## cnn_flower_pred
透過karas建立cnn辨識花的種類
## 開發環境

  * 開發環境 Visual Stdio 2022

## 專案需求說明

版本要求
  * Tensorflow 2.3.0 以上的版本適用
  * karas 2.3.1 以上的版本適用
  * numpy 1.9.1 以上的版本適用
  
## 資料集來源

* [資料集來源](https://www.kaggle.com/alxmamaev/flowers-recognition)

## 資料集說明

  * 訓練數據集包含 4242 張花卉圖像。
  * 每個檔案大約有 800 張照片(共五個檔案)。大約320x240像素。
  * 因照片大小不同，下載下來的資料集先做resize。
  
## 檔案說明

  * flowers資料夾：由資料集下載而來
  * newpic資料夾：經過resize後的圖片，作為模型訓練資料集
  * cnn.py 訓練模型、預測花的種類之主程式檔案
  * model.h5 模型檔
  * resize.py 重建資料集圖片大小

## 程式流程

1.引入函式庫  
2.定義參數  
3.資料前處理  
4.資料正規化(0到1之間)  
5.cnn模型建構  
 輸入為320X240  
 第一層卷積  
  filters=32  
  kernel=3  
  activation=relu  
 第二層卷積  
  filters=32  
  kernel=3  
  activation=relu  
 第三層卷積  
  filters=32  
  kernel=3  
  activation=relu  
 全連接層  
 神經元數=128  
 activation=relu  
6.模型編譯  
7.模型訓練  
8.儲存模型  
9.隨機取樣(10筆資料)  
10.預測花的種類  
  
