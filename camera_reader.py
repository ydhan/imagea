# -*- coding:utf-8 -*-
import cv2

from boss_train import Model
from image_show import show_image


if __name__ == '__main__':
    cap = cv2.VideoCapture("./7.jpeg")
    #cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
    cascade_path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
    model = Model()
    model.load()
    while True:
        _, frame = cap.read()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cascade = cv2.CascadeClassifier(cascade_path)
        print('start detected')
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        #facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=3, minSize=(3, 3))
        print("facerec = %d",len(facerect))
         
        if len(facerect) > 0:
            print('face detected')
            color = (255, 255, 255)  # 白
            for rect in facerect:
                #cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)

                x, y = rect[0:2]
                width, height = rect[2:4]
                image = frame[y - 10: y + height, x: x + width]

                #result = model.predict(image)
                result = model.predict(frame)
                if result == 0:  # boss
                    print('Boss is approaching')
                    show_image()
                else:
                    print('Not boss')

        k = cv2.waitKey(100)
        k=27
        if k == 27:
            break

    cap.release()
    #cv2.destroyAllWindows()
