import os
import cv2
import torch
from tqdm import trange
from mtcnn.mtcnn import MTCNN
from face.linknet import LinkNet34
import torchvision.transforms as transforms


def number_of_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def extract_and_write_face(video_path, write_dir, img_size, T=100, skin=False):
    cap = cv2.VideoCapture(video_path)
    if T == 0:
        T = number_of_frames(video_path)

    if skin:
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225],
                                                         inplace=False)])
        linknet = LinkNet34()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        linknet.load_state_dict(torch.load('face/model/linknet.pth', map_location=device))
        linknet.eval()

    face_detector = MTCNN()

    rot = 0
    ret, frame = cap.read()
    for tries in range(3):
        rows, cols, _ = frame.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rot, 1)
        dst = cv2.warpAffine(frame, M, (cols, rows))
        img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        dims = face_detector.detect_faces(img)
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
        dets = face_detector.detect_faces(img)
        if len(dets) > 0:
            x, y, w, h = dets[0]['box']
            face_image_true = img[y:y+h, x:x+w]
            face_h, face_w, c = face_image_true.shape
            face_image = cv2.resize(face_image_true, (img_size, img_size))
            if skin:
                trans_images = trans(face_image)
                skin_mask = linknet(trans_images.unsqueeze(0)).permute(0, 2, 3, 1).squeeze(0).data.cpu().numpy()
                images = face_image * skin_mask
                images = cv2.resize(images, (face_w, face_h))
                images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
            else:
                images = cv2.cvtColor(face_image_true, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(write_dir, '{0}.png'.format(i)), images)
        ret, frame = cap.read()
