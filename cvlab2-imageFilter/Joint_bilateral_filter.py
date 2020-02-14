
import cv2
import  numpy as np
import datetime

def jbf(source,guide,size,sigma_f,sigma_g):
    starttime=datetime.datetime.now()
    # 联合双边滤波
    # source为输入图像,三维矩阵
    # guide为引导图像,三维矩阵
    # size为滤波窗口大小
    # sigma_f为spatial kernel标准差
    # sigma_g为range kernel 标准差

    #返回处理后的图像的三维矩阵
    # Boundary是valid方式
    des=source.copy()

    distance = np.zeros([size, size], dtype=np.uint8)
    for m in range(size):
        for n in range(size):
            distance[m, n] = (m - size // 2) ** 2 + (n - size // 2) ** 2

    for i in range(size//2,guide.shape[0]-size//2):
        for j in range(size//2,guide.shape[1]-size//2):
            for d in range(3):
                #计算当前窗口范围
                istart = i - size//2
                iend = i+size//2
                jstart = j - size//2
                jend = j+size//2
                #原图的当前窗口
                window_s = source[istart:iend+1, jstart: jend+1, d]
                #引导图的当前窗口
                window_g = guide[istart:iend+1, jstart: jend+1, d]

                #由引导图像的灰度值差计算值域核
                g = np.exp(-0.5*(window_g - guide[i, j,d])**2 / (sigma_g **2))

                f=np.exp(-0.5*distance/(sigma_f**2))

                des[i,j,d]=int(np.sum(g*f*window_s)/np.sum(g*f))
    endtime = datetime.datetime.now()
    print('联合双边滤波操作时间：', endtime - starttime)
    return des









def zoom_bilinear_interpolation(source,rate):
    #rate 缩放比例，目标：原
    source = cv2.imread(source)
    starttime = datetime.datetime.now()
    if rate==1:
        return source.copy()

    x,y=int(source.shape[0]*rate),int(source.shape[1]*rate)
    des=np.zeros((x,y,3), dtype=np.uint8)

    for k in range(3):
        for i in range(x):
            for j in range(y):
                m = int(i/ rate)
                n =int(j / rate)
                xr = i/ rate - m
                yr = j / rate - n
                m1=min(m+1,source.shape[0]-1)
                n1=min(n+1,source.shape[1]-1)
                des[i,j,k] = source[m, n, k] * (1 - xr) * (1 - yr) + source[m1, n1, k] * xr * yr + source[m1, n, k] * xr * (1 - yr) + source[m, n1, k] * yr * (1 - xr)

    endtime = datetime.datetime.now()
    print('双线性缩放时间：',endtime-starttime)
    return des


if __name__ == '__main__':
    a=zoom_bilinear_interpolation('b.png',0.5)
    cv2.imwrite('b_half.png', a)
    guide=zoom_bilinear_interpolation('b_half.png',2)

    source = cv2.imread('b.png')

    des = jbf(source, guide, 5, 2, 10)

    # 按img_out路径保存图片，并显示两张前后对比图
    cv2.imwrite('b_pro.png', des)
    cv2.imshow('source(sigma_f={})+guide(sigma_g={})+result filter_size={}'.format(2, 10, 5),
               np.hstack([source[:guide.shape[0], :guide.shape[1], :], guide, des]))
    cv2.waitKey()
    cv2.destroyAllWindows()

    des=jbf(source,guide,17,3,3)

    #按img_out路径保存图片，并显示两张前后对比图
    cv2.imwrite('b_pro.png', des)
    cv2.imshow('source(sigma_f={})+guide(sigma_g={})+result filter_size={}'.format(3,3,17),np.hstack([source [:guide.shape[0],:guide.shape[1],:], guide , des]))
    cv2.waitKey()
    cv2.destroyAllWindows()

    des = jbf(source, guide, 5, 1, 1)

    # 按img_out路径保存图片，并显示两张前后对比图
    cv2.imwrite('b_pro.png', des)
    cv2.imshow('source(sigma_f={})+guide(sigma_g={})+result filter_size={}'.format(1,1,5), np.hstack([source[:guide.shape[0], :guide.shape[1], :], guide, des]))
    cv2.waitKey()
    cv2.destroyAllWindows()




