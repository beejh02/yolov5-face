import os
import cv2
import numpy as np
from pathlib import Path
from Crypto.Cipher import ChaCha20

# 암호화와 동일한 키
KEY = b'0123456789ABCDEF0123456789ABCDEF'  # 32 bytes

def decrypt_region(region: np.ndarray, key: bytes, nonce: bytes) -> np.ndarray:
    cipher = ChaCha20.new(key=key, nonce=nonce)
    decrypted = cipher.decrypt(region.tobytes())
    return np.frombuffer(decrypted, dtype=region.dtype).reshape(region.shape)

def decrypt_faces_in_video(video_path: str, face_npz_path: str, output_path: str):
    face_data = np.load(face_npz_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ 비디오 열기 실패: {video_path}")
        return

    # 비디오 정보 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # 무손실 코덱
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    print(f"🎞️ 복호화 시작: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_key = f"frame_{frame_idx}"
        if frame_key in face_data:
            boxes = face_data[frame_key]
            for (x1, y1, x2, y2) in boxes:
                nonce = frame_idx.to_bytes(12, byteorder='big')
                encrypted_region = frame[y1:y2, x1:x2].copy()
                decrypted_region = decrypt_region(encrypted_region, key=KEY, nonce=nonce)
                frame[y1:y2, x1:x2] = decrypted_region

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ 복호화 완료: {output_path}")

if __name__ == '__main__':
    # 파일 경로
    input_video = 'runs/detect/exp/output.avi'
    face_npz = 'runs/detect/exp/faces.npz'
    output_video = 'runs/detect/exp/decrypted.avi'

    decrypt_faces_in_video(input_video, face_npz, output_video)
