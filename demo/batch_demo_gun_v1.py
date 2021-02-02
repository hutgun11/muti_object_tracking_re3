import cv2
import glob
import numpy as np
import sys
import os.path
import os
import pandas as pd
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker
import xml.etree.cElementTree as ET

import natsort 

# if not os.path.exists(os.path.join(basedir, 'data')):
#     import tarfile
#     tar = tarfile.open(os.path.join(basedir, 'data.tar.gz'))
#     tar.extractall(path=basedir)
image_paths = natsort.natsorted(glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*.jpg')))#path data
image_read =cv2.imread(image_paths[0])
width=image_read.shape[1]
height=image_read.shape[0]
depth=image_read.shape[2]
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Image', 640, 480)
cv2.resizeWindow('Image', width, height)
classname=['tea plus','tea plus_2','rootbeer','blackpink can'] # rootbeer = 0 , tea_plus = 1 #key #edit
tracker = re3_tracker.Re3Tracker()
# image_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*.jpg')))#path data
# image_paths = natsort.natsorted(glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*.jpg')))#path data

# print(image_paths)

initial_bbox = [173, 93, 248, 292] # mouse #edit
initial_bbox2 =[263, 91, 331, 282] #mouse #edit
initial_bbox3 = [355, 65, 408, 275] # mouse #edit
initial_bbox4 =[427, 148, 478, 265] #mouse #edit

# Provide a unique id, an image/path, and a bounding box. 
#index เรียงตามนี้
tracker.track('tea plus', image_paths[0], initial_bbox)#key #edit
tracker.track('tea plus_2', image_paths[0], initial_bbox2)#key #edit
tracker.track('rootbeer', image_paths[0], initial_bbox3)#key #edit
tracker.track('blackpink can', image_paths[0], initial_bbox4)#key #edit

# print('ball track started')
li={'fullpath':[],'filename':[],'class':[],'x1':[],'y1':[],'x2':[],'y2':[]}
for ii,image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    # #gunfix
    imageRGB = image[:,:,::-1]

    bboxes = tracker.multi_track(['tea plus', 'tea plus_2','rootbeer','blackpink can'], imageRGB)#key #ใส่ list classname แทนเลย
    for bb,bbox in enumerate(bboxes):
        # print('coco',ii,classname[bb],bbox,image_path)
        color = cv2.cvtColor(np.uint8([[[bb * 255 / len(bboxes), 128, 200]]]),
            cv2.COLOR_HSV2RGB).squeeze().tolist()
        cv2.putText(image,'class :'+classname[bb],(int(bbox[0]), int(bbox[1])-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        cv2.rectangle(image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color, 2)#(x1,y1),(x2,y2)
        head_tail=os.path.split(image_path)
        folder=head_tail[0]
        filename=head_tail[1]

        li['fullpath'].append(image_path)
        li['filename'].append(filename)
        li['class'].append(classname[bb])
        li['x1'].append(int(bbox[0]))
        li['y1'].append(int(bbox[1]))
        li['x2'].append(int(bbox[2]))
        li['y2'].append(int(bbox[3]))

   
    # print('cola',ii,image_path, (int(bbox[0]), (int(bbox[1]), (int(bbox[2]), (int(bbox[3])))
    # print('cola',image_path,int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
    cv2.imshow('Image', image)  
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        # print(li)
        break


#df
df=pd.DataFrame(li)
# df.to_csv('test.csv',index=False)




#xml
last_row=len(df)-1
for index,row in df.iterrows():
        # print('index',index)
        if len(df['class'][index].split("_")) >1:
            class_name=df['class'][index].split("_")[0]
            # print('class_name1',class_name)
        else:
            class_name=df['class'][index]
            # print('class_name2',class_name)


        if index == 0: # start
            #create xml 
            annotation=ET.Element("annotation")
            ET.SubElement(annotation,'folder').text='data'
            ET.SubElement(annotation,"filename").text=df['filename'][index]
            ET.SubElement(annotation,"path").text='fixpath'
            source_tag=ET.SubElement(annotation,"source")
            ET.SubElement(source_tag,"database").text='Unknown'
            size=ET.SubElement(annotation,"size")
            ET.SubElement(size,"width").text=str(width)
            ET.SubElement(size,"height").text=str(height)
            ET.SubElement(size,"depth").text=str(depth)
            object_tag=ET.SubElement(annotation,"object")
            ET.SubElement(object_tag,"name").text=class_name
            ET.SubElement(object_tag,"pose").text='Unspecified'
            ET.SubElement(object_tag,"truncated").text='0'
            ET.SubElement(object_tag,"difficult").text='0'
            #bndbox
            bndbox=ET.SubElement(object_tag,"bndbox")
            ET.SubElement(bndbox,"xmin").text=str(df['x1'][index])
            ET.SubElement(bndbox,"ymin").text=str(df['y1'][index])
            ET.SubElement(bndbox,"xmax").text=str(int(df['x2'][index])-int(df['x1'][index]))
            ET.SubElement(bndbox,"ymax").text=str(int(df['y2'][index])-int(df['y1'][index]))

            
        elif index==last_row:
            #write xml
            object_tag=ET.SubElement(annotation,"object")
            ET.SubElement(object_tag,"name").text=class_name
            ET.SubElement(object_tag,"pose").text='Unspecified'
            ET.SubElement(object_tag,"truncated").text='0'
            ET.SubElement(object_tag,"difficult").text='0'
            #bndbox
            bndbox=ET.SubElement(object_tag,"bndbox")
            ET.SubElement(bndbox,"xmin").text=str(df['x1'][index])
            ET.SubElement(bndbox,"ymin").text=str(df['y1'][index])
            ET.SubElement(bndbox,"xmax").text=str(int(df['x2'][index])-int(df['x1'][index]))
            ET.SubElement(bndbox,"ymax").text=str(int(df['y2'][index])-int(df['y1'][index]))
            tree=ET.ElementTree(annotation)
            tree.write(df['fullpath'][index-1].replace('.jpg','.xml'))
            #write xml

        
        else:
            if df['filename'][index-1]==df['filename'][index]:
                object_tag=ET.SubElement(annotation,"object")
                ET.SubElement(object_tag,"name").text=class_name
                ET.SubElement(object_tag,"pose").text='Unspecified'
                ET.SubElement(object_tag,"truncated").text='0'
                ET.SubElement(object_tag,"difficult").text='0'
                #bndbox
                bndbox=ET.SubElement(object_tag,"bndbox")
                ET.SubElement(bndbox,"xmin").text=str(df['x1'][index])
                ET.SubElement(bndbox,"ymin").text=str(df['y1'][index])
                ET.SubElement(bndbox,"xmax").text=str(int(df['x2'][index])-int(df['x1'][index]))
                ET.SubElement(bndbox,"ymax").text=str(int(df['y2'][index])-int(df['y1'][index]))
            else:
                #write xml
                tree=ET.ElementTree(annotation)
                tree.write(df['fullpath'][index-1].replace('.jpg','.xml'))
                #write xml
                annotation=ET.Element("annotation")
                ET.SubElement(annotation,'folder').text='data'
                ET.SubElement(annotation,"filename").text=df['filename'][index]
                ET.SubElement(annotation,"path").text='fixpath'
                source_tag=ET.SubElement(annotation,"source")
                ET.SubElement(source_tag,"database").text='Unknown'
                size=ET.SubElement(annotation,"size")
                ET.SubElement(size,"width").text=str(width)
                ET.SubElement(size,"height").text=str(height)
                ET.SubElement(size,"depth").text=str(depth)
                object_tag=ET.SubElement(annotation,"object")
                ET.SubElement(object_tag,"name").text=class_name
                ET.SubElement(object_tag,"pose").text='Unspecified'
                ET.SubElement(object_tag,"truncated").text='0'
                ET.SubElement(object_tag,"difficult").text='0'
                #bndbox
                bndbox=ET.SubElement(object_tag,"bndbox")
                ET.SubElement(bndbox,"xmin").text=str(df['x1'][index])
                ET.SubElement(bndbox,"ymin").text=str(df['y1'][index])
                ET.SubElement(bndbox,"xmax").text=str(int(df['x2'][index])-int(df['x1'][index]))
                ET.SubElement(bndbox,"ymax").text=str(int(df['y2'][index])-int(df['y1'][index]))

#xml   