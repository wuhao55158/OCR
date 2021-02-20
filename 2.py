import cv2
import paddlehub as hub
# test_img_path = [r"D:\OR\test\3.png"]
# # 读取测试文件夹test.txt中的照片路径
# np_images =[cv2.imread(image_path) for image_path in test_img_path]
test_img_path = r"D:\OR\test\3.png"
# 读取测试文件夹test.txt中的照片路径C
#np_images =[cv2.imread(image_path) for image_path in test_img_path]
np_images=cv2.imread(test_img_path)

# 加载移动端预训练模型
ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
# 服务端可以加载大模型，效果更好
# ocr = hub.Module(name="chinese_ocr_db_crnn_server")
results = ocr.recognize_text(
                    images=np_images,         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                    use_gpu=False,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                    output_dir=r'D:\OR\test\ocr_result',  # 图片的保存路径，默认设为 ocr_result；
                    visualization=True,       # 是否将识别结果保存为图片文件；
                    box_thresh=0.5)          # 识别中文文本置信度的阈值；

for result in results:
    data = result['data']
    save_path = result['save_path']
    for infomation in data:
        print('text: ', infomation['text'], '\nconfidence: ', infomation['confidence'], '\ntext_box_position: ', infomation['text_box_position'])