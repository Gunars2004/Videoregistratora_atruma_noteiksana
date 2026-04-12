import cv2
import pytesseract
import os
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_FOLDER = "./data/slaidzinasana_image/"
OUTPUT_FILE = "./data/real_speed.txt"

def numerical_sort(filename):
    return int(os.path.splitext(filename)[0])

image_files = sorted(
    [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")],
    key=numerical_sort
)

labels = []
last_valid = 0

print("Processing images...")

for i, img_name in enumerate(image_files):
    frame = cv2.imread(os.path.join(IMAGE_FOLDER, img_name))
    h, w, _ = frame.shape

    # 🔥 paņem tikai apakšu
    roi = frame[int(h*0.92):h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # kontrasts
    gray = cv2.convertScaleAbs(gray, alpha=2.5, beta=0)

    # threshold
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)

    # ====== atrodam "KM" ======
    text = pytesseract.image_to_string(thresh, config='--psm 6')

    # ja neatrod KM → fallback
    if "KM" not in text:
        labels.append(last_valid)
        continue

    # ====== ņem tikai daļu pirms KM ======
    parts = text.split("KM")[0]

    # izvelk skaitļus
    digits = re.findall(r'\d+', parts)

    if digits:
        speed = int(digits[-1])  # pēdējais skaitlis pirms KM
        last_valid = speed
    else:
        speed = last_valid

    labels.append(speed)

    if i % 100 == 0:
        print(f"{i}/{len(image_files)} → {speed}")

with open(OUTPUT_FILE, "w") as f:
    for val in labels:
        f.write(str(val) + "\n")

print("DONE ")