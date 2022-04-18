from concurrent.futures.process import EXTRA_QUEUED_CALLS
import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt 

from IPython.display import clear_output,  display, HTML

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.models.modnet import MODNet

print(os.getcwd())

torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
output_videopath = 'result.mp4'
target_videopath = 'videos/target.mp4'

os.chdir('F:/Documents/github/MODNet')
cap_trainee = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap_coach = cv2.VideoCapture(target_videopath)
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

def crop_resize(img, crop):
    scale = 3
    h, w = img.shape[:2]
    if crop:
        cropped = img[300:h-150, 350:w-300]
    else:
        cropped = img
    result = cv2.resize(cropped, None, fx=scale, fy=scale)
    return result

def simple_merge_center(front, backgrond, mask=0, enhance=False, crop=False):
    
    ### CLIP
    mask = cv2.resize(mask, [front.shape[1], front.shape[0]])
    front = crop_resize(front, crop)
    mask = crop_resize(mask, crop)

    ###

    rowf, colf = front.shape[:2]
    rowb, colb = backgrond.shape[:2]

    ratio = 1.0
    if rowf > rowb or colf > colb:
        ratio = min(rowb / rowf, colb/colf) * 1.0

    rz_front_img = cv2.resize(front, (int(colf*ratio), int(rowf*ratio)), interpolation=cv2.INTER_AREA)
    rz_front_a_img = cv2.resize(mask, (int(colf*ratio), int(rowf*ratio)), interpolation=cv2.INTER_CUBIC)
   
    if enhance:
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))

        H, S, V = cv2.split(cv2.cvtColor(np.uint8(rz_front_img), cv2.COLOR_RGB2HSV))
        eq_V = clahe.apply(V)
        eq_S = S
        #eq_S = clahe.apply(S)
        rz_front_img = cv2.cvtColor(cv2.merge([H, eq_S, eq_V]), cv2.COLOR_HSV2RGB)

    rowf, colf = rz_front_img.shape[:2]
    
    lt = int((rowb-rowf)/2)
    ld = int((rowb+rowf)/2)
    rt = int((colb-colf)/2)
    rd = int((colb+colf)/2)
    
    blend_img = rz_front_a_img * rz_front_img + (1-rz_front_a_img) * backgrond[lt:ld, rt:rd]

    backgrond[lt:ld, rt:rd] = blend_img

    return backgrond


from moviepy.editor import *
def process_audio(src, audio_src_video_file):
    video_clip = VideoFileClip(src)
    #audio_clip = AudioFileClip(audio_src_video_file)
    audio_clip = AudioFileClip(audio_src_video_file).subclip(5,20)
    audio = afx.audio_loop(audio_clip, duration=video_clip.duration)
    video_clip_new = video_clip.set_audio(audio)
    video_clip_new.write_videofile(output_videopath, audio_codec="aac")


def process_image(img_input, img_background, enhance, crop):
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

    img_output = simple_merge_center(img_input, img_background, matte_np, enhance, crop)
    return img_output

def start():
    #p_trainee = get_property_from_cap(cap_trainee)
    p_coach = get_property_from_cap(cap_coach)

    if not cap_trainee.isOpened() or not cap_coach.isOpened():
        print("-----------ERROR: video not exists------------")
        return

    trainee_w = int(cap_trainee.get(cv2.CAP_PROP_FRAME_WIDTH))
    trainee_h = int(cap_trainee.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framerate = 24
    duration = p_coach['duration']
    width = max(trainee_w, p_coach['width'])
    height = max(trainee_h, p_coach['height'])
    framenum = duration * framerate

    h, w = bgd_img.shape[:2]
    
    
    i = 0
    while (cap_trainee.isOpened()):
        frame = i % framenum
        cap_coach.set(cv2.CAP_PROP_POS_FRAMES, frame)

        rval, left_frame = cap_trainee.read()   # first frame
        rval, right_frame = cap_coach.read()

        left_frame_np = process_image(left_frame, bgd_img.copy(), True, False)
        right_frame_np = process_image(right_frame, bgd_img.copy(), False, True)

        full_frame = np.concatenate([left_frame_np, right_frame_np], axis=1)
        cv2.imshow('result', full_frame)

        rval, left_frame = cap_trainee.read()
        i += 1

        if 0xFF & cv2.waitKey(30) == 27:
         break

start()
cv2.destroyAllWindows()
cap_trainee.release()
cap_coach.release()
