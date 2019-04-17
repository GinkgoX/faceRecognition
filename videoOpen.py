#-*- coding: utf-8 -*-

import cv2
import sys
from PIL import Image

def CatchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)
 
    cap = cv2.VideoCapture(camera_idx)        
        
    while cap.isOpened():
        ok, frame = cap.read() 
        if not ok:            
            break                    

        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        

    cap.release()
    cv2.destroyAllWindows() 
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo("Vide read", int(sys.argv[1]))
