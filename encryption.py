import os
import cv2
import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(BASE_DIR, "./input")
enc_dir = os.path.join(BASE_DIR, "./output/enc")
dec_dir = os.path.join(BASE_DIR, "./output/dec")
os.makedirs(enc_dir, exist_ok=True)
os.makedirs(dec_dir, exist_ok=True)

x, y, w, h = 100, 50, 200, 100

valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
key = get_random_bytes(32)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(valid_exts):
        input_path = os.path.join(input_dir, filename)
        print(f"\n▶ 처리 중: {filename}")

        img = cv2.imread(input_path)
        if img is None:
            print("❌ 이미지 불러오기 실패:", input_path)
            continue

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

        img_dec = img_enc.copy()
        img_dec[y:y+h, x:x+w] = roi_decrypted

        dec_output_path = os.path.join(dec_dir, f"dec_{filename}")
        cv2.imwrite(dec_output_path, img_dec)
        print(f"✅ 복호화 저장: {dec_output_path}")

        print("🔑 Key:", key.hex())
        print("🔓 Nonce:", nonce.hex())
