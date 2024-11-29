import time
import threading
import numpy as np
import cv2

import os
import fnmatch
import shutil
import os.path as osp
import stat

def find_folder(directory, folder_name):
    for root, dirnames, filenames in os.walk(directory):
        if folder_name in dirnames:
            return os.path.join(root, folder_name)
    

# 定义计数子线程
def timer(interval):
    i=0
    path = find_folder('./', 'calibration_imgs')
    print(path)
    os.chmod(path, stat.S_IWRITE)
    if osp.isdir(path):
        shutil.rmtree(path)
    # 创建目标文件夹
    os.makedirs(path)

    while True:  # 无限计时
        time.sleep(interval)
        if(img_temp is not None):
            cv2.imwrite(path+'/'+str(i)+'.png',img_temp)
            i+=1
            print("save!",i)


"""---------------- 主线程(main) -------------------"""
if __name__ == '__main__':
    img_temp = None  # 占位用的，目的是提升frame的作用域
    interval = 0.5  # 时间间隔(s)

    # 开启一个子线程
    """
        Note:
            1. daemon=
                1. True: 主线程结束，子线程也结束
                2. False：主线程结束，子线程不结束（主线程需等待子线程结束后再结束）
            2. args=(interval, )中的 逗号 不能省略（因为传入的必须是一个tuple）
    """
    # 1. 定义线程
    th1 = threading.Thread(target=timer, daemon=True, args=(interval,))
    # 2. 开启线程
    th1.start()

    # 创建摄像头对象
    cap=cv2.VideoCapture(1)
    time.sleep(1)


    width = 1920
    height = 1080
    fps = 60
    cap.set(3, width)  #设置宽度
    cap.set(4, height)  #设置长度
    cap.set(5, fps)  
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    width = cap.get(3)
    height  = cap.get(4)
    fps  = cap.get(5)
    Fourcc  = cap.get(6)

    print(width ,height, fps, Fourcc)

    while True:
        
        ret2 ,frame = cap.read()
        # 赋值变量
        img_temp = frame  # 将frame赋给img_temp
        cv2.imshow("capture2", frame)
        k=cv2.waitKey(1)
        if k==27:
            break
        

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
