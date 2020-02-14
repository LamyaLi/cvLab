import cv2
import datetime
import math

def image_warping(source):
    # source是已经读取好的表示图像的三维数组
    starttime = datetime.datetime.now()

    size_w,size_h=source.shape[0],source.shape[1]
    des=source.copy()

    for i in range(size_w):
        for j in range(size_h):
            i_n = (i-0.5*size_w)/(0.5*size_w)
            j_n = (j-0.5*size_h)/(0.5*size_h)

            r = (i_n**2+j_n**2)**0.5

            if r >= 1.0:
                i_s=i
                j_s=j
            else:
                theta =(1-r)**2
                i_n_s = math.cos(theta)*i_n-math.sin(theta)*j_n
                j_n_s = math.sin(theta)*i_n+math.cos(theta)*j_n
                i_s = i_n_s*0.5*size_w+0.5*size_w
                j_s = j_n_s*0.5*size_h+0.5*size_h

            m = int(i_s)
            n = int(j_s)
            xr = i_s - m
            yr = j_s - n
            m1 = min(m + 1, size_w - 1)
            n1 = min(n + 1, size_h - 1)
            des[i, j, :] = source[m, n, :] * (1 - xr) * (1 - yr) + source[m1, n1, :] * xr * yr + source[
                m1, n, :] * xr * (1 - yr) + source[m, n1, :] * yr * (1 - xr)

    endtime = datetime.datetime.now()
    print('图像扭曲时间：', endtime - starttime)
    return des


if __name__ == '__main__':

    source = cv2.imread('lab2.png')
    des=image_warping(source)

# 保存图片，并显示
    cv2.imwrite('lab2_warped.png', des)
    cv2.imshow('lab2_warped',des)
    cv2.waitKey()
    cv2.destroyAllWindows()