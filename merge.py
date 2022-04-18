import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.models.modnet import MODNet


torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

os.chdir("/home/anchen/demo/modnet/MODNet/")

cap_trainee = cv2.VideoCapture('videos/WeChat_20220415201156.mp4')
cap_coach = cv2.VideoCapture('videos/target.mp4')
bgd_img = cv2.imread('./black.png')

### load model
pretrained_ckpt = './pretrained/modnet_webcam_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

GPU = True if torch.cuda.device_count() > 0 else False
if GPU:
    print('Use GPU...')
    modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
else:
    print('Use CPU...')
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))

modnet.eval()


def get_property_from_cap(capture):
    res = {}
    res['width'] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    res['height'] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    res['framerate'] = int(capture.get(cv2.CAP_PROP_FPS))
    res['framenum'] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    res['duration'] = res['framenum'] / res['framerate'] # seconds
    return res


def simple_merge_center(front, backgrond, mask=0, enhance=False):
    rowf, colf = front.shape[:2]
    rowb, colb = backgrond.shape[:2]

    ratio = 1.0
    if rowf > rowb or colf > colb:
        ratio = min(rowb / rowf, colb/colf) * 1.0

    rz_front_img = cv2.resize(front, (int(colf*ratio), int(rowf*ratio)), interpolation=cv2.INTER_AREA)
    rz_front_a_img = cv2.resize(mask, (int(colf*ratio), int(rowf*ratio)), interpolation=cv2.INTER_CUBIC)
   
    if enhance:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

        H, S, V = cv2.split(cv2.cvtColor(np.uint8(rz_front_img), cv2.COLOR_RGB2HSV))
        eq_V = clahe.apply(V)
        rz_front_img = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)

    rowf, colf = rz_front_img.shape[:2]
    
    lt = int((rowb-rowf)/2)
    ld = int((rowb+rowf)/2)
    rt = int((colb-colf)/2)
    rd = int((colb+colf)/2)
    
    blend_img = rz_front_a_img * rz_front_img + (1-rz_front_a_img) * backgrond[lt:ld, rt:rd]
    '''
    if enhance:
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))

        hsv = cv2.cvtColor(np.uint8(blend_img), cv2.COLOR_RGB2HSV)

        hsv[...,0] = (np.uint8(hsv[...,0]))
        hsv[...,1] = clahe.apply(np.uint8(hsv[...,1]))
        hsv[...,2] = clahe.apply(np.uint8(hsv[...,2]))
        img_adj = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        backgrond[lt:ld, rt:rd] = img_adj
    else:
        backgrond[lt:ld, rt:rd] = blend_img
    
    '''
    backgrond[lt:ld, rt:rd] = blend_img

    return backgrond

def process_image(img_input, img_background, enhance):
    h, w = img_input.shape[:2]
    if w >= h:
        rh = 512
        rw = int(w / h * 512)
    else:
        rw = 512
        rh = int(h / w * 512)
    rh = rh - rh % 32
    rw = rw - rw % 32

    frame_np = cv2.resize(img_input, (rw, rh), cv2.INTER_AREA)

    frame_PIL = Image.fromarray(frame_np)
    frame_tensor = torch_transforms(frame_PIL)
    frame_tensor = frame_tensor[None, :, :, :]
    if GPU:
        frame_tensor = frame_tensor.cuda()

    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, True)

    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)

    img_output = simple_merge_center(img_input, img_background, matte_np, enhance)
    return img_output

def start():
    p_trainee = get_property_from_cap(cap_trainee)
    p_coach = get_property_from_cap(cap_coach)

    if not cap_trainee.isOpened() or not cap_coach.isOpened():
        print("-----------ERROR: video not exists------------")
        return

    framerate = p_trainee['framerate']
    duration = min(p_trainee['duration'], p_coach['duration'])
    width = max(p_trainee['width'], p_coach['width'])
    height = max(p_trainee['height'], p_coach['height'])
    framenum = duration * framerate

    h, w = bgd_img.shape[:2]

    print("Framerate %i, duration %i， width %i, height %i" %(framerate, duration, width, height))

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('result.mp4', fourcc, framerate, (w*2, h))
    
    rval, left_frame = cap_trainee.read()   # first frame
    
    with tqdm(range(int(framenum)))as t:
        for c in t:
            cap_coach.set(cv2.CAP_PROP_POS_FRAMES, int(c/p_trainee['framerate']*p_coach['framerate']))
            rval, right_frame = cap_coach.read()

            left_frame_np = process_image(left_frame, bgd_img.copy(), True)
            right_frame_np = process_image(right_frame, bgd_img.copy(), False)

            full_frame = np.concatenate([left_frame_np, right_frame_np], axis=1)
            video_writer.write(full_frame)

            rval, left_frame = cap_trainee.read()
            c += 1

            if c > 1200:
                video_writer.release()
                return

    video_writer.release()

start()