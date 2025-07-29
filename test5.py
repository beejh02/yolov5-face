import argparse
import time
from pathlib import Path
import sys
import os

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
    imgsz = check_img_size(img_size, s=model.stride.max())

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_video_path = str(Path(save_dir) / "output.avi")

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    dataset = LoadStreams(source, img_size=imgsz) if webcam else LoadImages(source, img_size=imgsz)
    bs = 1

    model.eval()

    all_face_data_np = {}
    video_writer = None
    frame_size_initialized = False

    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        if isinstance(im0s, list):
            img0 = im0s[0].copy()
        else:
            img0 = im0s.copy()

        img = letterbox(img0, new_shape=imgsz)[0]
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.inference_mode():
            pred = model(img)[0]
            pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        im0 = img0.copy()
        face_data = []

        for det in pred:
            if len(det):
                boxes = scale_coords(img.shape[2:], det[:, :4].clone(), im0.shape).round()
                det = det.clone()
                det[:, :4] = boxes

                for j in range(det.size(0)):
                    xyxy = det[j, :4].view(-1).tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    face_region = im0[y1:y2, x1:x2].copy()

                    nonce = frame_idx.to_bytes(12, byteorder='big', signed=False)
                    enc_region = encrypt_region(face_region, key=KEY, nonce=nonce)
                    im0[y1:y2, x1:x2] = enc_region

                    face_data.append([x1, y1, x2, y2])

        if not frame_size_initialized:
            h, w = im0.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w, h))
            frame_size_initialized = True

        video_writer.write(im0)

        frame_key = f"frame_{frame_idx}"
        face_array = np.array(face_data, dtype=np.uint16) if face_data else np.empty((0, 4), dtype=np.uint16)
        all_face_data_np[frame_key] = face_array

        if frame_idx > 0 and frame_idx % 240 == 0:
            npz_path = Path(save_dir) / "faces.npz"
            np.savez(npz_path, **all_face_data_np)

    if video_writer:
        video_writer.release()

    if len(all_face_data_np) > 0:
        npz_path = Path(save_dir) / "faces.npz"
        np.savez(npz_path, **all_face_data_np)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5n-face.pt')
    parser.add_argument('--source', type=str, default='0')
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
