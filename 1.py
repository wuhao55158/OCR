import cv2
import paddlehub as hub
test_img_path = r"C:\Users\byw\Desktop\新建文件夹\a.png"

##https://blog.csdn.net/qq_44486439/article/details/109698115
# 读取测试文件夹test.txt中的照片路径C
#np_images =[cv2.imread(image_path) for image_path in test_img_path]
#np_images=cv2.imread(test_img_path)
# import matplotlib.image as mpimg
# np_images=[mpimg.imread('a.png')]
np_images=[cv2.imread('a.jpg')]
# 加载移动端预训练模型
#ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
# 服务端可以加载大模型，效果更好
ocr = hub.Module(name="chinese_ocr_db_crnn_server")
results = ocr.recognize_text(
                    images=np_images,         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                    use_gpu=False,          # 识别中文文本置信度的阈值；
                    box_thresh = 0.5,  # 检测文本框置信度的阈值；
                    text_thresh = 0.5)  # 识别中文文本置信度的阈值；

for result in results:
    data = result['data']
    #save_path = result['save_path']
    for infomation in data:
        #print('text: ', infomation['text'])
        print(infomation['text'])
        #print('text: ', infomation['text'], '\nconfidence: ', infomation['confidence'], '\ntext_box_position: ', infomation['text_box_position'])