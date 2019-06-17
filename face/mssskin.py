import logging
import os
import pickle
import time
from copy import copy
from multiprocessing import Process, Queue

import cv2
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import torch
from mssskin import (DETECTION_FOREHEAD, DETECTION_MANUAL, DETECTION_SKIN,
                     MODEL_PATH)
from face.face_tracking import get_forehead
from face.model import Model
from face.preprocess import VIDEO_SIZE, preprocess_image
from face.signal_process import detrend
from face.skin_detection import skin_detection
from scipy.interpolate import interp1d
from torch.autograd import Variable

from tqdm import trange

torch.set_num_threads(6)

MODEL_URL = 'https://github.com/ml-lab/Multiscale-Super-Spectral/tree/master/TestLog'

def generate_signal(mss, skin_pixels):
    m = interp1d([380, 740], [0, 31])
    # Lenghtwaves ranges
    r_range = range(int(m(380)), int(m(521)))
    g_range = range(int(m(521)), int(m(625)))
    b_range = range(int(m(626)), int(m(740)))
    r = np.mean([mss[0, r_range, i, j]
                 for (i, j) in zip(skin_pixels[0], skin_pixels[1])])
    g = np.mean([mss[0, g_range, i, j]
                 for (i, j) in zip(skin_pixels[0], skin_pixels[1])])
    b = np.mean([mss[0, b_range, i, j]
                 for (i, j) in zip(skin_pixels[0], skin_pixels[1])])
    return r, g, b


def generate_signal_rgb(rgb, skin_pixels):
    r = np.mean([rgb[0, 0, i, j]
                 for (i, j) in zip(skin_pixels[0], skin_pixels[1])])
    g = np.mean([rgb[0, 1, i, j]
                 for (i, j) in zip(skin_pixels[0], skin_pixels[1])])
    b = np.mean([rgb[0, 2, i, j]
                 for (i, j) in zip(skin_pixels[0], skin_pixels[1])])
    return r, g, b

def number_of_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def extract_and_write_face(video_path, write_dir, T=100):
    cap = cv2.VideoCapture(video_path)
    if T == 0:
        T = number_of_frames(video_path)

    for i in trange(T):
        ret, frame = cap.read()

        rows, cols, _ = frame.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
        dst = cv2.warpAffine(frame, M, (cols, rows))

        dims = get_forehead(dst)
        if dims.any():
            x, y, w, h = dims[0]
            forehead_img = dst[y:y+h, x:x+h]

            cv2.imwrite(os.path.join(write_dir, '{0}.png'.format(i)), forehead_img)

def frame_producer(video_path, detection, max_frames, q):
    if detection == DETECTION_MANUAL:
        skin_pixels_manual = select_roi(video_path)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frames = 1
    while cap.isOpened() and frames < max_frames:
        ret, frame = cap.read()
        frame = cv2.resize(frame, VIDEO_SIZE, interpolation=cv2.INTER_AREA)
        if detection == DETECTION_FOREHEAD:
            skin_pixels = get_forehead(frame)
        elif detection == DETECTION_SKIN:
            skin_pixels = skin_detection(frame)
        else:
            skin_pixels = skin_pixels_manual
        img = preprocess_image(frame)
        img = torch.Tensor(img)
        img = torch.unsqueeze(img, 0)
        img = Variable(img)
        q.put((img, skin_pixels))
        frames += 1
    q.put(None)
    while q.qsize() > 0:
        pass


def select_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame = cv2.resize(frame, VIDEO_SIZE, interpolation=cv2.INTER_AREA)
    r = cv2.selectROI(frame)
    return (range(int(r[1]), int(r[1]+r[3])),
            range(int(r[0]), int(r[0]+r[2])))


def process_signal(signal):
    signal = detrend(signal, 10)
    return signal
#    return lfilter(firwin(8, 0.3),
#                   np.ones(len(signal)),
#                   signal)


def frame_consumer(queue, oqueue, signal, max_frames, model):
    msg = queue.get()
    while msg:
        img, skin_pixels = msg
        if model:
            with torch.no_grad():
                hyper = np.array(model.forward(img))
            s_r, s_g, s_b = generate_signal(hyper, skin_pixels)
        else:
            s_r, s_g, s_b = generate_signal_rgb(img, skin_pixels)
        oqueue.put((s_r, s_g, s_b))
        msg = queue.get()
    oqueue.put(None)


def plot_signal(ax, lines, signal):
    signal_view = copy(signal)
    for i in range(3):
        signal_view[i] = process_signal(np.array(signal_view[i]))
        lines[i].set_xdata(range(len(signal_view[i])))
        lines[i].set_ydata(signal_view[i])
    ax.set_xlim(0, len(signal_view[1]))

    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)


def process_image(img):
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
    model = load_model()
    real_rgb = torch.Tensor(preprocess_image(img))
    real_rgb = torch.unsqueeze(real_rgb, 0)
    real_rgb = Variable(real_rgb)
    with torch.no_grad():
        f = model.forward(real_rgb)
    return img, f


def load_model():
    rgb_features = 3
    hyper_features = 31
    negative_slope = 0.2
    dropout = 0
    model = Model(
        input_features=rgb_features,
        output_features=hyper_features,
        negative_slope=negative_slope,
        p_drop=dropout
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model).to(device)
    model_name = 'Model'
    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_PATH, 'net_%s_%02d.pth'%(model_name, 0)),
            map_location=device))
    model.eval()
    return model
