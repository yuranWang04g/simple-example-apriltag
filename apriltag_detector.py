# coding: UTF-8
import apriltag
import cv2

font=cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag25h9') )
i=0
while(1):
    # 获得图像
    ret, frame = cap.read()
    # 检测按键
    k=cv2.waitKey(1)
    #print(k)
    if k==27:
        break
    #elif k==ord('s'):
        #cv2.imwrite('/OpenCV_pic/'+str(i)+'.png', frame)
        #i+=1
    # 检测apriltag
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray)
    """ for tag in tags:
        n=tag.tag_id
        cv2.circle(frame, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2) # left-top
        cv2.circle(frame, tuple(tag.corners[1].astype(int)), 4, (0, 255, 0), 2) # right-top
        cv2.circle(frame, tuple(tag.corners[2].astype(int)), 4, (0, 0, 255), 2) # right-bottom
        cv2.circle(frame, tuple(tag.corners[3].astype(int)), 4, (255, 255, 0), 2) # left-bottom
        l=tag.center.astype(int)
        cv2.putText(frame,("%d"%n),l,font,2,(255,0,255),3) """

    # 显示检测结果
    #cv2.namedWindow("capture",0)
    #cv2.resizeWindow("capture",400,300)
    cv2.imshow('capture', frame)
#print(tags)
cap.release()
cv2.destroyAllWindows()
