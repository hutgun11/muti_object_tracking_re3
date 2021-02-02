import numpy as np
import cv2
import math
import os
from PIL import Image

path_video='video/IMG_9500.MOV'
classname='tea_plus_rootbeer'
def resize(image):
    w_resize =576
    (img_h,img_w,_)=image.shape
    if img_w > w_resize:
        r =w_resize/img_w
        dim=(w_resize,int(img_h*r))
        image =cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    return image
def cut_frame(path_video,classname):
    vidcap =cv2.VideoCapture(path_video)
    success,image = vidcap.read()
    seconds=0.2
    fps =vidcap.get(cv2.CAP_PROP_FPS)
    multiplier=int(fps*seconds)

    while success:
        frameId =int(round(vidcap.get(1)))
        success,image=vidcap.read()
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE) #mobile case
        image=resize(image)
        path_output="data/"+classname+str(frameId)+".jpg"
        if frameId % multiplier == 0 :
            cv2.imwrite(path_output,image)
            print(frameId,path_output)
cut_frame(path_video,classname)