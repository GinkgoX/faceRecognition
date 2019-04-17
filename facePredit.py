#-*- coding: utf-8 -*-

import cv2
import sys
import gc
from trainKersa import Model

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
        
    model = Model()		#load model
    model.load_model(file_path = 'model.h5')        
    color = (0, 255, 0)
    cap = cv2.VideoCapture(int(sys.argv[1]))
    cascade_path = "haarcascade_frontalface_alt2.xml"    #path for human face classificer
    
    #predict face
    while True:
        _, frame = cap.read()  
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_path)                
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect

                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)   

                if faceID == 0:                                                        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame,'personA', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)                                 
                if faceID == 1:                                                        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame,'personB', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)                                        
                if faceID == 2:                                                        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame,'personBaba', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)               

                if faceID == 3:                                                        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame,'..', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                if faceID == 4:                                                        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame,'..', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)                      
                else:
                    pass
        cv2.imshow("Who am I", frame)
        k = cv2.waitKey(10)
        if k & 0xFF == ord('q'):		#exit the project
            break
            
    cap.release()		#release camera
    cv2.destroyAllWindows()
    
    
