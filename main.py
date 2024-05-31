import cv2
import os
from ultralytics import YOLO

model = YOLO("best.pt")
image_dir = "C:/Users/90545/Desktop/image"

if not os.path.exists(image_dir):
    print(f"{image_dir} dizini bulunamadı.")
    exit()

image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir) if image_name.endswith(('.png', '.jpg', '.jpeg'))]

if not image_paths:
    print(f"{image_dir} dizininde imaj bulunamadı.")
    exit()

for frame_path in image_paths:

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"{frame_path} imajı okunamadı. Atlanıyor.")
        continue

    sonuclar = model.track(frame, persist=True, conf=0.55)

    işaretlenmiş_imaj = sonuclar[0].plot()

    cv2.imshow("YOLOv8 Takip", işaretlenmiş_imaj)

    tuş = cv2.waitKey(0)

    if tuş == ord('q'):
        break
cv2.destroyAllWindows()