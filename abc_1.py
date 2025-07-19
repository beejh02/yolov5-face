import argparse
import time
from pathlib import Path
import sys
import os

import numpy as np
import cv2
import torch
import copy

from Crypto.Cipher import ChaCha20
from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, scale_coords, increment_path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

KEY = b'0123456789ABCDEF0123456789ABCDEF'
NONCE = b'12345678ABCD'

def encrypt_region(region: np.ndarray, key: bytes = KEY, nonce: bytes = NONCE) -> np.ndarray:
    plaintext = region.tobytes()
    cipher = ChaCha20.new(key=key, nonce=nonce)
    ciphertext = cipher.encrypt(plaintext)
    enc_region = np.frombuffer(ciphertext, dtype=region.dtype).reshape(region.shape)
    return enc_region


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)
    return model


def detect(model, source, device, project, name, exist_ok, save_img, view_img):
    img_size = 480
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz = (img_size, img_size)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    if webcam:
        dataset = LoadStreams(source, img_size=imgsz)
        bs = 1
    else:
        dataset = LoadImages(source, img_size=imgsz)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    for path, im, im0s, vid_cap in dataset:
        orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis=0) if len(im.shape) == 4 else im.transpose(1, 2, 0)
        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]
        r = img_size / max(h0, w0)

        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())
        img = letterbox(img0, new_shape=imgsz)[0]
        img = img.transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')

        for i, det in enumerate(pred):
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)
            save_path = str(Path(save_dir) / p.name)

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for j in range(det.size(0)):
                    xyxy = det[j, :4].view(-1).tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    face_region = im0[y1:y2, x1:x2].copy()

                    enc_region = encrypt_region(face_region)
                    im0[y1:y2, x1:x2] = enc_region

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    try:
                        vid_writer[i].write(im0)
                    except Exception as e:
                        print(e)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5n-face.pt')
    parser.add_argument('--source', type=str, default='./myimages')
    parser.add_argument('--img-size', type=int, default=480)
    parser.add_argument('--project', default=ROOT / 'runs/detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--save-img', default=True, action='store_true')
    parser.add_argument('--view-img', action='store_true')
    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(opt.weights, device)
    detect(model, opt.source, device, opt.project, opt.name, opt.exist_ok, opt.save_img, opt.view_img)
    end_time = time.time()
    print(f"\n총 실행 시간: {end_time - start_time:.2f}초")
