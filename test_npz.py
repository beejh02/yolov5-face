import argparse
import time
from pathlib import Path
import sys
import os
import json

import numpy as np
import cv2
import torch

from Crypto.Cipher import ChaCha20
from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, scale_coords, increment_path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

KEY = b'0123456789ABCDEF0123456789ABCDEF'  # 32 bytes (256-bit key)

def encrypt_region(region: np.ndarray, key: bytes, nonce: bytes) -> np.ndarray:
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

    all_face_data_np = {}

    for path, im, im0s, vid_cap in dataset:
        orgimg = im0s if isinstance(im0s, np.ndarray) else im0s.copy()
        img0 = orgimg

        imgsz = check_img_size(img_size, s=model.stride.max())
        img = letterbox(img0, new_shape=imgsz)[0]
        img = img.transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        print(len(pred[0]), 'face(s)')

        for i, det in enumerate(pred):
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)
            save_path = str(Path(save_dir) / p.name)

            face_data = []

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for j in range(det.size(0)):
                    xyxy = det[j, :4].view(-1).tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    face_region = im0[y1:y2, x1:x2].copy()

                    nonce = frame.to_bytes(12, byteorder='big', signed=False)
                    enc_region = encrypt_region(face_region, key=KEY, nonce=nonce)
                    im0[y1:y2, x1:x2] = enc_region

                    face_data.append([x1, y1, x2, y2])

            frame_key = f"frame_{frame}"
            face_array = np.array(face_data, dtype=np.uint16) if face_data else np.empty((0, 4), dtype=np.uint16)
            all_face_data_np[frame_key] = face_array

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

    if len(all_face_data_np) > 0:
        npz_path = str(Path(save_dir) / f"{p.stem}_faces.npz")
        np.savez(npz_path, **all_face_data_np)


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
