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

# 경로 설정
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)  # YOLOv5 root directory
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, scale_coords, increment_path


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def show_results(img, xyxy, conf, class_num, filename):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])

    x = x1; y = y1; w = x2 - x1; h = y2 - y1
    
    roi = img[y:y+h, x:x+w].tobytes()
    nonce = (int(filename.replace(".jpg", ""))).to_bytes(8)
    
    cipher = ChaCha20.new(key=key, nonce=nonce)
    encrypted = cipher.encrypt(roi)

    img_enc = img.copy()
    roi_encrypted = np.frombuffer(encrypted, dtype=np.uint8).reshape((h, w, 3))
    img_enc[y:y+h, x:x+w] = roi_encrypted

    enc_output_path = os.path.join(enc_dir, f"enc_{filename}")
    cv2.imwrite(enc_output_path, img_enc)
    print(f"✅ 암호화 저장: {enc_output_path}")


    cipher_dec = ChaCha20.new(key=key, nonce=nonce)
    decrypted = cipher_dec.decrypt(encrypted)
    roi_decrypted = np.frombuffer(decrypted, dtype=np.uint8).reshape((h, w, 3))

    # 복호화된 ROI 삽입
    img_dec = img_enc.copy()
    img_dec[y:y+h, x:x+w] = roi_decrypted

    # 복호화 이미지 저장
    dec_output_path = os.path.join(dec_dir, f"dec_{filename}")
    cv2.imwrite(dec_output_path, img_dec)
    print(f"✅ 복호화 저장: {dec_output_path}")


    print("🔑 Key:", key.hex())
    print("🔓 Nonce:", nonce.hex())
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)
    return img


def detect(model, source, device, project, name, exist_ok, save_img, view_img):
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz = (640, 640)

    # 결과 저장 디렉토리 생성
    save_dir = increment_path(os.path.join(project, name), exist_ok=exist_ok)
    os.makedirs(save_dir, exist_ok=True)

    is_file = os.path.splitext(source)[1][1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    # Dataloader
    if webcam:
        print('loading streams:', source)
        dataset = LoadStreams(source, img_size=imgsz)
        bs = 1
    else:
        print('loading images', source)
        dataset = LoadImages(source, img_size=imgsz)
        bs = 4
    vid_path, vid_writer = [None] * bs, [None] * bs

    for path, im, im0s, vid_cap in dataset:
        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis=0)
        else:
            orgimg = im.transpose(1, 2, 0)

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

            filename = os.path.basename(p)
            save_path = os.path.join(save_dir, filename)

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    class_num = det[j, 15].cpu().numpy()
                    im0 = show_results(im0, xyxy, conf, class_num, filename)

            if view_img:
                cv2.imshow('result', im0)
                cv2.waitKey(1)

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
                        save_path = os.path.splitext(save_path)[0] + '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    try:
                        vid_writer[i].write(im0)
                    except Exception as e:
                        print(e)


if __name__ == '__main__':
    start_time = time.time()
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    key = get_random_bytes(32)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    enc_dir = os.path.join(BASE_DIR, "./output/enc")
    dec_dir = os.path.join(BASE_DIR, "./output/dec")

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5n-face.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./myimages', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--project', default=os.path.join(ROOT, 'runs', 'detect'), help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-img', default=True, action='store_true', help='save results')
    parser.add_argument('--view-img', action='store_true', help='show results')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    detect(model, opt.source, device, opt.project, opt.name, opt.exist_ok, opt.save_img, opt.view_img)

    end_time = time.time()
    print(f"\n총 실행 시간: {end_time - start_time:.2f}초")
