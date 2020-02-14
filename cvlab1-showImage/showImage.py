import cv2
import matplotlib.pyplot as plt
import pyglet
import numpy as np

#静态图片用opencv的imread读取
img1=cv2.imread('Img1.png')
img2=cv2.imread('Img2.jpg')
img3=cv2.imread('Img3.bmp')

#通过打印内容及shape属性，可以理解图片读入为三维数组，以BGR顺序
print(img3)
print(img3.shape,img3.size,img3.dtype,type(img3))

#opencv的imshow函数仅能展示单张图片，或多张size一致的图片
cv2.imshow('image', img1)
cv2.waitKey()
cv2.destroyAllWindows()

#通过matplotlib的figure窗口将三张size不一静态图片一起显示
#先将三张静态图片组合成一个思维数组，经验证未损失改变图像信息
imgs = [img1,img2,img3]
print(img3==imgs[2])
plt.figure()
for i in range(0,3):

    b, g, r = cv2.split(imgs[i])
    rgb_img = cv2.merge([r, g, b])
    plt.subplot(1,3,i+1)
    plt.imshow(rgb_img)
plt.show()

#利用pyglet展示gif图像


# 在工作目录中选择一个gif动画文件
ag_file = "Img4.gif"
animation = pyglet.resource.animation(ag_file)
sprite = pyglet.sprite.Sprite(animation)
# 创建一个窗口并将其设置为图像大小
win = pyglet.window.Window(width=sprite.width, height=sprite.height)
# 设置窗口背景颜色 = r, g, b, alpha
# 每个值从 0.0 到 1.0
green = 0, 1, 0, 1
pyglet.gl.glClearColor(*green)
@win.event
def on_draw():
    win.clear()
    sprite.draw()
pyglet.app.run()

#读入四通道图像，打印a.shape显示a为四维数组
a=cv2.imread('a.png',cv2.IMREAD_UNCHANGED)
print(a.shape)

#选取alpha通道，并展示
a_alpha=a[:,:,3]
cv2.imshow('image',a_alpha)
cv2.waitKey()
cv2.destroyAllWindows()

#更改背景并展示新图像
bg=cv2.imread('bg.png')
for i in range(a_alpha.shape[0]):
    for j in range(a_alpha.shape[1]):
        if a_alpha[i, j] == 0:
            a[i, j, 0:3] = bg[i, j, 0:3]

cv2.imshow('image',a)
cv2.waitKey()
cv2.destroyAllWindows()



