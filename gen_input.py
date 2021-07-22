import cv2
import numpy as np

net_wh = (224, 224)

img_path = "dataset/img/ILSVRC2012_val_00000003.JPEG"

orig_image = cv2.imread(img_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, net_wh)
image = (image - 127.5) / 1
image = np.expand_dims(image, axis=0)
image = image.astype(np.int8)
out_file = "dataset/input.bin"
image.tofile(out_file)
print(f"save to {out_file} OK")
