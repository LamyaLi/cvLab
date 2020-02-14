import cv2
import matplotlib.pyplot as plt
import  numpy as np
from skimage import measure,color

def iseyes(img_two, minr, minc, maxr, maxc):
    ##如果区域内有两个以上的空框是眼睛
    part = np.zeros(((maxr - minr), (maxc - minc)))

    for i in range(minr, maxr):
        for j in range(minc, maxc):
            if img_two[i, j] == 0:
                part[i - minr, j - minc] = 255
            else:
                part[i - minr, j - minc] = 0

    part_labeled, num = measure.label(part, return_num=True, connectivity=1)  ##八邻域

    global img
    img_copy=img.copy()
    count=0
    for region2 in measure.regionprops(part_labeled):
        min_row2, min_col2, max_row2, max_col2 = region2.bbox
        w=max_col2-min_col2
        h=max_row2-min_row2
        total_w=maxc-minc
        total_h=maxr-minr
        w_ratio=w/total_w
        h_ratio=h/total_h
        if w_ratio<1/3 and h_ratio<0.2 and w_ratio>0.05 and h_ratio>1/30 and w>=h:
            count=count+1
            img_copy = cv2.rectangle(img_copy, (min_col2 + minc, min_row2 + minr), (max_col2 + minc, max_row2 + minr),(0, 255, 0), 2)
    print(count)
    if count>=1:
        img=img_copy
        return True
    return False

#######################################################################################begin
img = cv2.imread('Orical1.jpg')
#img = cv2.imread('Orical2.jpg')

img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

plt.imshow(img_ycrcb)
plt.show()

y,cr,cb=cv2.split(img_ycrcb)
cr_gaussian=cv2.GaussianBlur(cr,(5,5),0)
cb_gaussian=cv2.GaussianBlur(cb,(5,5),0)

skin=np.zeros_like(cr)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if y[i][j]<70:
            skin[i][j] = 0
        elif cr_gaussian[i][j]>133 and cr_gaussian[i][j]<173 and cb_gaussian[i][j]>77 and cb_gaussian[i][j]<127:
            skin[i][j]=255
        else:
            skin[i][j]=0

plt.imshow(skin)
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
skin_opening = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
plt.imshow(skin_opening)
plt.show()
print(skin_opening.shape)

skin_labeled= measure.label(skin_opening,connectivity = 2)##八邻域
dst=color.label2rgb(skin_labeled)
plt.imshow(dst)
plt.show()

count_face=0
for region in measure.regionprops(skin_labeled):

    min_row, min_col, max_row, max_col=region.bbox

    if (max_row - min_row)/img.shape[1]>1/15 and (max_col - min_col)/img.shape[0]>0.05:
        height_width_ratio = (max_row - min_row) / (max_col - min_col)
        if height_width_ratio>0.6 and height_width_ratio<2.0:
            if iseyes(skin_opening,min_row, min_col, max_row, max_col):
            #print(height_width_ratio)
                count_face = count_face+1
                img = cv2.rectangle(img, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)

print("face!!!",count_face)

cv2.imwrite('after.jpg', img)
cv2.imshow('face recognition', img)
cv2.waitKey()
cv2.destroyAllWindows()
