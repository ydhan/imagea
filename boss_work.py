# -*- coding:utf-8 -*-
import cv2

from boss_train import Model
from image_show import show_image


if __name__ == '__main__':
    
    model = Model()
    model.load()
    
    frame = cv2.imread("./dog.jpg",3)
    #cv2.imshow(frame)
    result = model.predict(frame)
    if result == 0:  # boss
        print('Boss is approaching')
	#show_image()
    else:
        print('Not boss')
