import numpy as np
import torch
import cv2
import sys
import onnxruntime
from blazebase import resize_pad, denormalize_detections
from visualization import draw_detections,  draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS


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

onnx_file_name = 'Preprocess1x256x256xBGRxByte.onnx'
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_profiling = True
ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options)

input_name = ort_session.get_inputs()[0].name
    
while hasFrame:
    img1, img2, scale, pad = resize_pad(frame)

    img_in = np.expand_dims(img1, axis=0).astype(np.uint8)
    ort_inputs = {input_name: img_in}

    ort_outs = ort_session.run(None, ort_inputs)

    imgDisp = ort_outs[0][0]

    cv2.imshow(WINDOW, imgDisp)

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
