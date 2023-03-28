訓練數據集包含 4242 張花卉圖像。
每個檔案大約有 800 張照片(共五個檔案)。大約320x240像素。
因照片大小不同，下載下來的資料集先做resize。
資料總量為4242*320*240

resize.py程式流程
1.引入函式庫
2.定義參數
3.主程式執行
4.定義Resize_Image function

cnn.py
程式流程：
1.引入函式庫
2.定義參數
3.資料前處理
4.資料正規化(0到1之間)
5.cnn模型建構
輸入為320*240
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
9.隨機取樣(10比資料)
10.預測花的種類

