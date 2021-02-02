
#draw box
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle.
ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,boxDraw

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),2)
            a=x
            b=y
            if a != x | b != y:
                pass
                    # cv2.rectangle(img,(ix,iy),(x,y),(0,0,0),-1)
       

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),2)

    x1=ix
    y1=iy
    x2=x
    y2=y
    
    boxDraw=[x1,y1,x2,y2]

image_paths = natsort.natsorted(glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*.jpg')))#path data
image_read =cv2.imread(image_paths[0])
width=image_read.shape[1]
height=image_read.shape[0]
depth=image_read.shape[2]
img = image_read
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
# cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', width, height)
initial_bbox=[]
while(1):
    cv2.imshow('image',img)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        print('cola',boxDraw)
        initial_bbox.append(boxDraw)

        # mode = not mode
    elif k == 32:
        break
#draw box