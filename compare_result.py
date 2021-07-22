import numpy as np
import imagenet_classes as class_name

logits = np.fromfile("dataset/output.bin", dtype=np.int8)
# k = 5
# cls_idx = logits.argsort()[-k:][::-1]
cls = np.argmax(logits)
cls_names = np.asarray(class_name.class_names)
print("class is " + cls_names[cls])

