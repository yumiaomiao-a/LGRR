#_*_coding:utf-8 _*_
import os

import numpy as np
# import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import random
import torch
# from skimage import data, color



def region_label(input_img):
    # print('********',input_img.shape)

    # cv2.CascadeClassifier(r"/data/anaconda3/envs/ymmtf/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    face_detector = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    input_img = input_img.cpu().numpy()*255
    input_img = input_img.astype('uint8')
    input_img = np.transpose(input_img,(1,2,0))

    # plt.imshow(input_img)
    # plt.savefig('./input_img.jpg')

    gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2BGRA)
    # gray = color.rgb2gray(input_img)

    face = face_detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=3,minSize=(32,32), flags=4)#scaleFactor=1.05


    [img_h, img_w, img_c] = input_img.shape

    for (x, y, w, h) in face:
        # print('***********',x, y, w, h,input_img.shape)


        if x>30 and w>100:
            input_img[:,:,:]=0
            input_img[y+10:y+h-10, x+10:x+w-10, :] = 1    #255
            # input_img = cv2.blur(input_img,(6,6))

        else:
            input_img[:,:,:]=0
            input_img[50:200, 50:200, :] = 1    #255
            # input_img = cv2.blur(input_img,(10,10))


        # input_img[0:img_h, 0:x, :] = 0
        # input_img[0:img_h, x+w:img_w :] = 0
        # input_img[0:y, 0:img_w :] = 0
        # input_img[w+y:img_h, 0:img_w :] = 0


        region_label = input_img[...,::-1]

        if region_label is None:
            input_img = np.array([240,240,3])
            input_img = torch.from_numpy(input_img)

            input_img[:,:,:]=0
            input_img[50:200, 50:200, :] = 1
            input_img = cv2.blur(input_img,(6,6))
            region_label = input_img[...,::-1]



        # print('________________',face_eraser_gray.dtype)

        # plt.imshow(region_label)
        # plt.savefig('./fakelabel.jpg')


        region_label = region_label.astype('float32')
        region_label = torch.from_numpy(region_label)
        region_label = np.transpose(region_label,(2,0,1))
        # region_label = color.gray2rgb(region_label)
        # region_label = np.transpose(region_label,(2,0,1))
        # region_label = region_label.astype('float32')


        # print('_____________',face_eraser_gray.shape)
        # print('_____________',face_eraser_gray.dtype)
        # print('++++++++++++++++',type(region_label))
        return region_label



# def fakelabel_new():
#     input_img = np.zeros((240,240,3))
#     input_img[50:200, 50:200, :] = 0
#     region_label = input_img
#     region_label = np.transpose(region_label,(2,0,1))
#     region_label = region_label.astype('float32')
#     return region_label


def fakelabel_new():
    input_img = np.ones((240,240,3))

    input_img = input_img#*255
    # input_img[50:200, 50:200, :] = 0
    region_label = input_img
    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')
    return region_label

def fakelabel_new_30():
    input_img = np.ones((30,30,3))
    input_img = input_img*255
    # input_img[50:200, 50:200, :] = 0
    region_label = input_img
    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')
    return region_label

def fakelabel_new_60():
    input_img = np.ones((60,60,3))
    input_img = input_img*255
    # input_img[50:200, 50:200, :] = 0
    region_label = input_img
    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')
    return region_label

def fakelabel_new_15():
    input_img = np.ones((15,15,3))
    input_img = input_img*255
    # input_img[50:200, 50:200, :] = 0
    region_label = input_img
    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')
    return region_label


def fakelabel_new_240():
    input_img = np.ones((240,240,1))
    input_img = input_img*255
    # input_img[50:200, 50:200, :] = 0
    region_label = input_img
    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')
    return region_label

def fakelabel_new_120():
    input_img = np.ones((120,120,3))
    input_img = input_img*255
    # input_img[50:200, 50:200, :] = 0
    region_label = input_img
    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')
    return region_label



def reallabel_new():
    input_img = np.zeros((240,240,3))
    # input_img = input_img*255
    # input_img[0:31, 0:31, :] = 255
    # input_img[50:200, 50:200, :] = 255
    region_label = input_img
    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')
    return region_label

def reallabel_new_30():
    input_img = np.zeros((30,30,3))
    # input_img = input_img*255
    # input_img[0:31, 0:31, :] = 255
    # input_img[50:200, 50:200, :] = 255
    region_label = input_img
    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')
    return region_label

def reallabel_new_15():
    input_img = np.zeros((15,15,3))
    # input_img = input_img*255
    # input_img[0:31, 0:31, :] = 255
    # input_img[50:200, 50:200, :] = 255
    region_label = input_img
    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')
    return region_label

def reallabel_new_240():
    input_img = np.zeros((240,240,1))
    # input_img = input_img*255
    # input_img[0:31, 0:31, :] = 255
    # input_img[50:200, 50:200, :] = 255
    region_label = input_img
    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')
    return region_label


def reallabel_new_60():
    input_img = np.zeros((60,60,3))
    # input_img = input_img*255
    # input_img[0:31, 0:31, :] = 255
    # input_img[50:200, 50:200, :] = 255
    region_label = input_img
    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')
    return region_label

def reallabel_new_120():
    input_img = np.zeros((120,120,3))
    # input_img = input_img*255
    # input_img[0:31, 0:31, :] = 255
    # input_img[50:200, 50:200, :] = 255
    region_label = input_img
    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')
    return region_label






def region_reallabel_all(input_img):
    zeros = 255

    input_img = input_img.cpu().numpy()*255
    input_img = input_img.astype('uint8')
    input_img = np.transpose(input_img,(1,2,0))

    [img_h, img_w, img_c] = input_img.shape

    input_img[0:img_h, 0:img_w, :] = 255

    region_label = input_img[...,::-1]

    # region_label = region_label.astype('float32')
    # region_label = torch.from_numpy(region_label)
    # region_label = np.transpose(region_label,(2,0,1))

    region_label = cv2.cvtColor(region_label,cv2.COLOR_BGRA2RGB)
    # region_label = color.gray2rgb(region_label)

    # print('××××_________________',type(region_label))
    # print('×××××××××_________________',region_label.shape)

    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')

    # print('_________________',type(region_label))


    return region_label





def region_fakelabel_all(input_img):

    input_img = input_img.cpu().numpy()*255
    input_img = input_img.astype('uint8')
    input_img = np.transpose(input_img,(1,2,0))

    [img_h, img_w, img_c] = input_img.shape

    input_img[0:img_h, 0:img_w, :] = 255

    region_label = input_img[...,::-1]

    # region_label = region_label.astype('float32')
    # region_label = torch.from_numpy(region_label)
    # region_label = np.transpose(region_label,(2,0,1))

    region_label = cv2.cvtColor(region_label,cv2.COLOR_BGRA2RGB)
    # region_label = color.gray2rgb(region_label)

    # print('××××_________________',type(region_label))
    # print('×××××××××_________________',region_label.shape)

    region_label = np.transpose(region_label,(2,0,1))
    region_label = region_label.astype('float32')

    # print('_________________',type(region_label))
    return region_label



if __name__ == '__main__':
    frame = cv2.imread('real.jpg')