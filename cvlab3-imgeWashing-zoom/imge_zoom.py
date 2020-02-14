import cv2
import  numpy as np
import datetime

def zoom_bilinear_interpolation(source,x_rate,y_rate):
    #rate 缩放比例，目标：原
    #source是已经读取好的表示图像的三维数组
    starttime = datetime.datetime.now()
    if x_rate==1 and y_rate==1:
        return source.copy()

    x,y=int(source.shape[0]*x_rate),int(source.shape[1]*y_rate)
    des=np.zeros((x,y,3), dtype=np.uint8)

    for k in range(3):
        for i in range(x):
            for j in range(y):
                m = int(i/ x_rate)
                n =int(j / y_rate)
                xr = i/ x_rate - m
                yr = j / y_rate - n
                m1=min(m+1,source.shape[0]-1)
                n1=min(n+1,source.shape[1]-1)
                des[i,j,k] = source[m, n, k] * (1 - xr) * (1 - yr) + source[m1, n1, k] * xr * yr + source[m1, n, k] * xr * (1 - yr) + source[m, n1, k] * yr * (1 - xr)

    endtime = datetime.datetime.now()
    print('双线性缩放时间：',endtime-starttime)
    return des


if __name__ == '__main__':

    source = cv2.imread('lab2.png')
    des=zoom_bilinear_interpolation(source,2,2)

# 保存图片，并显示
    cv2.imwrite('lab2_zoomed.png', des)
    cv2.imshow('lab2_zoomed',des)
    cv2.waitKey()
    cv2.destroyAllWindows()