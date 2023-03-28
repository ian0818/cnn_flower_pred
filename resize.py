
import cv2
import os

image_path='d:/tf/CNNmodel/flowers/rose'
filename='rose'

def Resize_Image(image_path,w=320,h=240):
    image_path_list = os.listdir(image_path)
    image_path_list.sort()
    number=1
    for filename in image_path_list:
        file = image_path+'/'+filename
        img = cv2.imread((file), cv2.IMREAD_COLOR)
        # 调用cv2.resize函数resize图片
        new_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img_name = str(number)+'.jpg'
        number+=1
        # 生成圖片儲存路徑
        save_path = image_path + '_new/'
        if os.path.exists(save_path):
            print(number)
            # 調用cv.2的imwrite函數保存圖片
            save_img = save_path + img_name
            cv2.imwrite(save_img, new_img)
        else:
            os.mkdir(save_path)
            save_img = save_path + img_name
            cv2.imwrite(save_img, new_img)

Resize_Image(image_path)

