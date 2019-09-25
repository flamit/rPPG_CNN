import os
import cv2
import dlib
import torch
from tqdm import trange
from face.linknet import LinkNet34
import torchvision.transforms as transforms


def number_of_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def extract_and_write_face(video_path, write_dir, T=100, skin=False):
    cap = cv2.VideoCapture(video_path)
    if T == 0:
        T = number_of_frames(video_path)

    predictor_path = 'face/model/predictor.dat'
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    if skin:
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225],
                                                         inplace=False)])
        linknet = LinkNet34()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        linknet.load_state_dict(torch.load('face/model/linknet.pth', map_location=device))
        linknet.eval()

    rot = 90
    ret, frame = cap.read()
    for tries in range(3):
        rows, cols, _ = frame.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rot, 1)
        dst = cv2.warpAffine(frame, M, (cols, rows))
        img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        dims = detector(img, 1)
        if len(dims) == 0:
            print("Face not detected, rotating and trying again...")
            rot += 90
        else:
            print("Video rotation found to be {} degrees".format(rot))
            break
    if rot == 360:
        print("No face detected in the video")
        raise

    for i in trange(T):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rows, cols, _ = frame.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rot, 1)
        img = cv2.warpAffine(img, M, (cols, rows))
        dets = detector(img, 1)
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(img, detection))
        images = dlib.get_face_chips(img, faces, size=320)
        if skin:
            trans_images = trans(images[0])
            skin_mask = linknet(trans_images.unsqueeze(0)).permute(0, 2, 3, 1).data.cpu().numpy()
            skin_pixels = images * skin_mask
            images = cv2.cvtColor(skin_pixels[0], cv2.COLOR_RGB2BGR)
        else:
            images = cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(write_dir, '{0}.png'.format(i)), images)
        ret, frame = cap.read()
