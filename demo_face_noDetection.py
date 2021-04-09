import numpy as np
import torch
import cv2
import sys

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazepalm import BlazePalm
from blazeface_landmark import BlazeFaceLandmark
from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

face_regressor = BlazeFaceLandmark().to(gpu)
face_regressor.load_weights("blazeface_landmark.pth")


WINDOW='test'
cv2.namedWindow(WINDOW)
if len(sys.argv) > 1:
    capture = cv2.VideoCapture(sys.argv[1])
    mirror_img = False
else:
    capture = cv2.VideoCapture(2)
    mirror_img = True

if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False

while hasFrame:
    frame_ct +=1

    img1, img2, scale, pad = resize_pad(frame)
    img3 = cv2.resize(img1, (192,192))

    x = torch.from_numpy(img3).to(gpu)
    x = x.unsqueeze(0)

    flags, normalized_landmarks = face_regressor(x)

    for i in range(len(flags)):
        landmark, flag = normalized_landmarks[i], flags[i]
        if flag>.5:
            draw_landmarks(img3, landmark[:,:2], FACE_CONNECTIONS, size=1)

    cv2.imshow(WINDOW, img3)

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
