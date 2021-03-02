import numpy as np
import cv2
import math
import os
from PIL import Image
from datetime import datetime
# from videoprops import get_video_properties

folder='video/'
# folder_success='video_success/'
for subdir,dirs,files in os.walk(folder):
    for file in files:
        path_video=os.path.join(folder,file)
        # path_move = os.path.join(folder_success,file)
# filevideo='P_one2.mov'
# # path_video='video/IMG_9556.MOV'
# path_video=os.path.join(folder,filevideo)

# props_2 =get_video_properties(path_video)
# def cal_resolution(props_2):
#     rotate=False
#     width =props_2['width']
#     height=props_2['height']
#     if width < height:
#         reso = width/height
#     else:
#         reso = height/width
#         rotate=True
#     if reso == 0.5625:
#         resize_v=576
#     else:
#         resize_v=768
#     return resize_v,rotate
# resize_v,rotate=cal_resolution(props_2)

classname=file.strip('mp4MOVmov.')+'_'+datetime.today().strftime('%Y_%m_%d')+'_'

def resize(image):
    # w_resize =768#for 3:4
    w_resize =576#for 9:16
    # w_resize=resize_v
    (img_h,img_w,_)=image.shape
    if img_w > w_resize:
        r =w_resize/img_w
        dim=(w_resize,int(img_h*r))
        image =cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    return image
def cut_frame(path_video,classname):
    vidcap =cv2.VideoCapture(path_video)
    success,image = vidcap.read()
    # seconds=0.2
    seconds=0.17
    fps =vidcap.get(cv2.CAP_PROP_FPS)
    multiplier=int(fps*seconds)

    while success:
        frameId =int(round(vidcap.get(1)))
        success,image=vidcap.read()
        # if rotate:
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE) #mobile case
        image=resize(image)
        path_output="data/"+classname+str(frameId)+".jpg"
        if frameId % multiplier == 0 :
            cv2.imwrite(path_output,image)
            print(frameId,path_output)
    return "success"
    # vidcap.()
def main_capture():
    if not os.path.exists('data'):
        os.makedirs('data')
    try:
        cut_frame(path_video,classname)
        if not os.path.exists('video_success'):
            os.makedirs('video_success')
    except:
        pass
    return path_video
# main_capture()