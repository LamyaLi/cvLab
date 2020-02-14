import cv2
import numpy as np
import math
import datetime
def Gaussian(img_input,img_output,sigma):
    #导入图像
    img_input=cv2.imread(img_input)

    starttime = datetime.datetime.now()

    img_input2=img_input.copy()
    img_input3 = img_input.copy()
    img_size=img_input.shape

    size = int(sigma * 6 - 1) // 2 * 2 + 1
    centre=size//2
    #Boundary是valid方式
    core=getGaussianCore(sigma)
    #行卷积
    for i in range(img_size[0]):
        for j in range(centre,img_size[1]-centre):
            for d in range(3):
                sum=0
                for k in range(size):
                    sum=sum+core[k]*img_input[i][j-centre+k][d]
                img_input2[i][j][d]=sum

    #列卷积
    for i in range(centre,img_size[0]-centre):
        for j in range(centre,img_size[1]-centre):
            for d in range(3):
                sum=0
                for k in range(size):
                    sum=sum+core[k]*img_input2[i-centre+k][j][d]
                img_input3[i][j][d]=sum

    endtime=datetime.datetime.now()
    print('运行时间：', endtime - starttime)

    #按img_out路径保存图片，并显示两张前后对比图
    cv2.imwrite(img_output, img_input3)
    cv2.imshow('GaussianFilter,sigma={}'.format(sigma),np.hstack([img_input,img_input3]))
    cv2.waitKey()
    cv2.destroyAllWindows()


def getGaussianCore(sigma):
    size = int(sigma * 6 - 1) // 2 * 2 + 1
    centre=size//2
    core= np.zeros([size], np.float32)
    coefficient=1.0/(sigma*((2*math.pi)**0.5))
    sum=0
    for i in range(size):
        core[i]=coefficient*math.exp(-0.5*((i-centre)/sigma)**2)
        sum=sum+core[i]

    #归一化
    core=core/sum
    return core


if __name__ == '__main__':

    Gaussian('a.jpg','a_pro.jpg',1)

    Gaussian('a.jpg','a_pro.jpg',3)
