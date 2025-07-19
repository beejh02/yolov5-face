# -*- coding: UTF-8 -*-
import argparse
import time
import sys
import os
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

import numpy as np
import cv2
import torch
import copy
import struct

# YOLOv5 루트 경로 설정
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, scale_coords, increment_path


def load_model(weights, device):
    return attempt_load(weights, map_location=device)


def encrypt_and_draw(img, xyxy, key, filename, face_idx):
    # 원본 이미지 크기
    H, W = img.shape[:2]

    # 좌표 정수화
    x1, y1, x2, y2 = map(int, xyxy)
    # 이미지 범위 안으로 클램핑
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))

    # 유효 크기 검사
    if x2 <= x1 or y2 <= y1:
        return None  # invalid

    w, h = x2 - x1, y2 - y1

    # ROI 바이트 추출 (BGR 그대로)
    roi = img[y1:y2, x1:x2].tobytes()

    # nonce: 파일명(숫자) → 8바이트 빅엔디안
    base = os.path.splitext(filename)[0]
    try:
        idx = int(base)
        nonce = struct.pack(">Q", idx)
    except ValueError:
        nonce = get_random_bytes(8)

    # 암호화
    cipher = ChaCha20.new(key=key, nonce=nonce)
    encrypted = cipher.encrypt(roi)
    enc_roi = np.frombuffer(encrypted, dtype=np.uint8).reshape((h, w, 3))

    # 복호화
    cipher_dec = ChaCha20.new(key=key, nonce=nonce)
    decrypted = cipher_dec.decrypt(encrypted)
    dec_roi = np.frombuffer(decrypted, dtype=np.uint8).reshape((h, w, 3))

    return (x1, y1, x2, y2), enc_roi, dec_roi


def detect(model, source, device, project, name, exist_ok, save_img, view_img, enc_dir, dec_dir, key):
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5

    save_dir = increment_path(os.path.join(project, name), exist_ok=exist_ok)
    os.makedirs(save_dir, exist_ok=True)

    is_file = os.path.splitext(source)[1][1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    dataset = LoadStreams(source, img_size=(img_size, img_size)) if webcam else LoadImages(source, img_size=(img_size, img_size))
    bs = 1 if webcam else 4
    vid_path, vid_writer = [None] * bs, [None] * bs

    for path, im, im0s, vid_cap in dataset:
        # im은 BGR 상태
        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0,2,3,1), axis=0)
        else:
            orgimg = im.transpose(1,2,0)

        img0 = copy.deepcopy(orgimg)  # BGR 그대로
        imgsz = check_img_size(img_size, s=model.stride.max())
        img = letterbox(img0, new_shape=(imgsz, imgsz))[0]
        img = img.transpose(2,0,1).copy()
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 추론 & 후처리
        pred = model(img)[0]
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            p, im0, frame = (path[i], im0s[i].copy(), dataset.count) if webcam else (path, im0s.copy(), getattr(dataset, 'frame', 0))
            filename = os.path.basename(p)

            # 암호화·복호화 베이스로 사용할 이미지 복사
            img_enc = im0.copy()
            img_dec = img_enc.copy()

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for j in range(det.size(0)):
                    xyxy = det[j, :4].tolist()
                    result = encrypt_and_draw(im0, xyxy, key, filename, j)
                    if result is None:
                        print(f"⚠️ invalid box for face {j}, skipping.")
                        continue

                    (x1,y1,x2,y2), enc_roi, dec_roi = result
                    # 암/복호화된 ROI 덮어쓰기
                    img_enc[y1:y2, x1:x2] = enc_roi
                    img_dec[y1:y2, x1:x2] = dec_roi
                    # 바운딩 박스 그리기
                    cv2.rectangle(im0, (x1,y1), (x2,y2), (0,255,0), 2, cv2.LINE_AA)

            # 저장
            enc_path = os.path.join(enc_dir, f"enc_{filename}")
            dec_path = os.path.join(dec_dir, f"dec_{filename}")
            cv2.imwrite(enc_path, img_enc)
            cv2.imwrite(dec_path, img_dec)
            print(f"✅ 암호화: {enc_path}")
            print(f"✅ 복호화: {dec_path}")

            # 디스플레이
            if view_img:
                cv2.imshow('result', im0)
                cv2.waitKey(1)
            # 결과 저장
            if save_img:
                out_path = os.path.join(save_dir, filename)
                cv2.imwrite(out_path, im0)


if __name__ == '__main__':
    start = time.time()
    key = get_random_bytes(32)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    enc_dir = os.path.join(BASE_DIR, "output/enc")
    dec_dir = os.path.join(BASE_DIR, "output/dec")
    os.makedirs(enc_dir, exist_ok=True)
    os.makedirs(dec_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/yolov5n-face.pt', nargs='+')
    parser.add_argument('--source', default='./myimages')
    parser.add_argument('--project', default=os.path.join(ROOT, 'runs', 'detect'))
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--save-img', action='store_true', default=True)
    parser.add_argument('--view-img', action='store_true')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    detect(model, opt.source, device, opt.project, opt.name, opt.exist_ok,
           opt.save_img, opt.view_img, enc_dir, dec_dir, key)

    print(f"\n총 실행 시간: {time.time() - start:.2f}초")
