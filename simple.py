# -*- coding:utf-8*-
import cv2
capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# 图像处理函数
def processImg(img):
    # 画出一个框
    cv2.rectangle(img, (500, 300), (800, 400), (0, 0, 255), 5, 1, 0)
    # 上下翻转
    # img= cv2.flip(img, 0)
    return img

# 一帧帧地show
while (capture.isOpened()):
    ret, frame = capture.read()
    if not ret:
        break

    result = processImg(frame)
    cv2.imshow('result', result)

    # esc键退出
    if 0xFF & cv2.waitKey(30) == 27:
        break

cv2.destroyAllWindows()
capture.release()
