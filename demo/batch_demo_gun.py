import cv2
import glob
import numpy as np
import sys
import os.path
import os
import pandas as pd
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
import xml.etree.cElementTree as ET
# from read_cap import *
# from tracker import re3_tracker
from tracker import re3_tracker
from read_cap import main_capture
from datetime import datetime
import natsort 
import shutil
tracker = re3_tracker.Re3Tracker()
# status_pause=True
path_video,path_move=main_capture()#capture video to images 
index=0
last_file=None
round_counter=1
start=True
status_pause=False
image_paths = natsort.natsorted(glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*.jpg')))#path data
threshold_iou=20
percent_sku=20
def check_size_sku(size_draw_box,w,h,bb,class_sku,percent_sku):
    if size_draw_box[bb]['class'] == class_sku:
        percent_width=size_draw_box[bb]['width']*(percent_sku/100)
        percent_height=size_draw_box[bb]['height']*(percent_sku/100)
        width1_range=int(size_draw_box[bb]['width']-percent_width)
        width2_range=int(size_draw_box[bb]['width']+percent_width)
        height1_range=int(size_draw_box[bb]['height']-percent_height)
        height2_range=int(size_draw_box[bb]['height']+percent_height)
        range_width=range(width1_range,width2_range)
        range_height=range(height1_range,height2_range)
        print(width1_range,w,width2_range,height1_range,h,height2_range,class_sku,size_draw_box[bb]['class'])
        
        if w in range_width and h in range_height:
            size_sku=True
        else:
            size_sku=False
    return size_sku,w in range_width, h in range_height
            
def cal_iou(bb1,bb2):
    x_left=max(bb1['x1'],bb2['x1'])
    y_top=max(bb1['y1'],bb2['y1'])
    x_right =min(bb1['x2'],bb2['x2'])
    y_bottom = min(bb1['y2'],bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area=(x_right - x_left)*(y_bottom-y_top)
    bb1_area = (bb1['x2']-bb1['x1'])*(bb1['y2']-bb1['y1'])
    bb2_area =(bb2['x2']-bb2['x1'])*(bb2['y2']-bb2['y1'])
    iou = intersection_area / float(bb1_area +bb2_area -intersection_area)
    return iou*100
    

def write_xml(li,width,height,depth):
    df=pd.DataFrame(li)
    # df.to_csv('test.csv',index=False)
    #xml
    last_row=len(df)-1
    for index,row in df.iterrows():
            # print('index',index)
            if len(df['class'][index].split("@")) >1:
                class_name=df['class'][index].split("@")[0]
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
                ET.SubElement(bndbox,"xmax").text=str(int(df['x2'][index]))
                ET.SubElement(bndbox,"ymax").text=str(int(df['y2'][index]))
                # ET.SubElement(bndbox,"xmax").text=str(int(df['x2'][index])-int(df['x1'][index]))
                # ET.SubElement(bndbox,"ymax").text=str(int(df['y2'][index])-int(df['y1'][index]))

                
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
                ET.SubElement(bndbox,"xmax").text=str(int(df['x2'][index]))
                ET.SubElement(bndbox,"ymax").text=str(int(df['y2'][index]))
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
                    ET.SubElement(bndbox,"xmax").text=str(int(df['x2'][index]))
                    ET.SubElement(bndbox,"ymax").text=str(int(df['y2'][index]))
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
                    ET.SubElement(bndbox,"xmax").text=str(int(df['x2'][index]))
                    ET.SubElement(bndbox,"ymax").text=str(int(df['y2'][index]))

def main(status_pause=False,index=0,last_file=None,start=True):
    '''
        read classname.txt to list
    '''
    with open('config/classname.txt') as f:
        classname =f.read().splitlines()
    counter_class=0


    # status_pause=False
    #draw box
    drawing = False # true if mouse is pressed
    ix,iy = -1,-1

    # mouse callback function
    def draw_bndbox(event,x,y,flags,param):
        global ix,iy,drawing,mode,boxDraw

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                # cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
                a=x
                b=y
                if a != x | b != y:
                    pass
                        # cv2.rectangle(img,(ix,iy),(x,y),(0,0,0),-1)
        

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)

        x1=ix
        y1=iy
        x2=x
        y2=y
        
        boxDraw=[x1,y1,x2,y2]

        # print(boxDraw)

    # image_paths = natsort.natsorted(glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*.jpg')))#path data
    print('pause',status_pause,last_file)
    if last_file!=None :#status_pause=True
        # print('last_file_draw',last_file)
        # image_read =cv2.imread(last_file)
        print('last_file_draw',image_paths[index],last_file)
        image_read =cv2.imread(last_file)
    else:
        print('first_file_draw',image_paths[0])
        image_read =cv2.imread(image_paths[0])
    width=image_read.shape[1]
    height=image_read.shape[0]
    depth=image_read.shape[2]
    img = image_read
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_bndbox)
    # cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', width, height)
    initial_bbox=[]
    size_draw_box=[]
    while(1):
        cv2.imshow('image',img)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            print('Drawbox :',boxDraw,'class :',classname[counter_class])
            # print('width',boxDraw[2]-boxDraw[0],'height',boxDraw[3]-boxDraw[1],'class',classname[counter_class])
            try:
                print('nextclass :'+classname[counter_class+1])
            except:
                print("label complete")
                pass
            counter_class+=1
            initial_bbox.append(boxDraw)
            # print('width',boxDraw[2]-boxDraw[0],'height',boxDraw[3]-boxDraw[1])
            try:
                size_draw_box.append({'width':boxDraw[2]-boxDraw[0],'height':boxDraw[3]-boxDraw[1],'class':classname[counter_class-1]})
            except:
                pass
            if len(classname)==counter_class:
                break
            # mode = not mode
        elif k == 32:#space bar for object tracking
            break
        elif cv2.waitKey(1) & 0xFF ==ord('q'):
            start=False
            sys.exit(0)
            break
            # cv2.destroyAllWindows()

    #draw box





























    #track object


    # print('size_draw_box',size_draw_box)
    width=image_read.shape[1]
    height=image_read.shape[0]
    depth=image_read.shape[2]
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', width, height)


    #index เรียงตามนี้
    for index_class in range(len(classname)):
        if status_pause:
            
            tracker.track(classname[index_class],image_paths[index],initial_bbox[index_class])
            print('track_name_pause:',image_paths[index])

        else:
            tracker.track(classname[index_class],image_paths[0],initial_bbox[index_class])
            print('track_name:',image_paths[0])


    li={'fullpath':[],'filename':[],'class':[],'x1':[],'y1':[],'x2':[],'y2':[]}
    for ii,image_path in enumerate(image_paths[index:]):
        print('track_imread',image_path,index,ii,index+ii)
        image = cv2.imread(image_path)
        # #gunfix
        imageRGB = image[:,:,::-1]

        bboxes = tracker.multi_track(classname, imageRGB)#key #ใส่ list classname แทนเลย
        for bb,bbox in enumerate(bboxes):
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
            #check size sku
            w=int(bbox[2]-bbox[0])
            h=int(bbox[3]-bbox[1])
            class_sku=classname[bb]
            
            # print('bbox',int(bbox[2]-bbox[0]),int(bbox[3]-bbox[1]),classname[bb])
            '''
            check size sku
            '''
            status_size_sku,status_w,status_h=check_size_sku(size_draw_box,w,h,bb,class_sku,percent_sku)
            if status_size_sku == False:
                status_pause=True
                ii=ii+index
                image_path=image_path
                write_xml(li,width,height,depth)
                # percent_sku +=2
                print('***************size_sku_False','status_weight',status_w,'status_height',status_h,class_sku)
                return status_pause,ii,image_path,start
            '''
            check overlap by iou
            '''
            if bb==0:
                bb1={'x1':bbox[0],'y1':bbox[1],'x2':bbox[2],'y2':bbox[3],'class':classname[bb]}
            else:
                bb2={'x1':bbox[0],'y1':bbox[1],'x2':bbox[2],'y2':bbox[3],'class':classname[bb]}
            if bb != 0:
                try:
                    iou=cal_iou(bb1,bb2)
                except:
                    iou=0
                    pass
                if iou >threshold_iou : #threshold
                    status_pause=True
                    ii=ii+index
                    image_path=image_path
                    write_xml(li,width,height,depth)
                    print('***************iou',iou,bb1['class'],bb2['class'])
                    return status_pause,ii,image_path,start
                else:
                    bb1=bb2
        cv2.imshow('Image', image)  
        status_pause=False
        #pause function
        if cv2.waitKey(1) & 0xFF ==ord('p'):
            status_pause=True
            ii=ii+index
            image_path=image_path
            write_xml(li,width,height,depth)
            print('pause_gun',ii,classname[bb],bbox,image_path)
            
            break
            # return pause,ii,image_path
        #kill function
        elif cv2.waitKey(1) & 0xFF ==ord('q'):
            sys.exit(0)
            start=False
            break
    write_xml(li,width,height,depth)
    if status_pause :
        start=True
    else:
        start=False
    
    
    return status_pause,ii,image_path,start


# if cut_video :
#     cut_video=False
#     cut_frame(path_video,classname)
#     if not os.path.exists('video_success'):
#         os.makedirs('video_success')
#     shutil.move(path_video,path_move)
    # cut_video=False
while start:
    print('round:',round_counter,status_pause,index,last_file)
    status_pause,index,last_file,start=main(status_pause,index,last_file,start)
    round_counter+=1
    cv2.destroyAllWindows()
    

