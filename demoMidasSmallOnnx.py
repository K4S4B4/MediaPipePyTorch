import numpy as np
import torch
import cv2
import sys
from blazebase import resize_pad, denormalize_detections
from visualization import draw_detections,  draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS
import onnxruntime


def draw_landmarks(img, points, shiftY, connections=[], color=(0, 255, 0), size=2):
    #points = points[:,:2]
    for point in points:
        x, y = point
        x, y = int(x), int(y) + shiftY
        cv2.circle(img, (x, y), size, color, thickness=size)
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        x0, y0 = int(x0), int(y0) + shiftY
        x1, y1 = int(x1), int(y1) + shiftY
        cv2.line(img, (x0, y0), (x1, y1), (0,0,0), size)

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

#onnx_file_name = 'resource/MiDaS/model-small.onnx'
#input_size = 256

onnx_file_name = 'resource/MiDaS/model-f6b98070.onnx'
input_size = 384

sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_profiling = True
ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options)

input_name = ort_session.get_inputs()[0].name
    
while hasFrame:
    img1, img2, scale, pad = resize_pad(frame)

    img1 = cv2.resize(img1, (input_size,input_size))


    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0

    img_in = np.transpose(img1, (2, 0, 1))
    img_in = np.ascontiguousarray(img_in).astype(np.float32)

    img_in = np.expand_dims(img_in, axis=0).astype(np.float32)

    ort_inputs = {input_name: img_in}

    ort_outs = ort_session.run(None, ort_inputs)

    out = ort_outs[0][0] * 255/ 7000
    #depth = ort_outs[0][0]

    #depth_min = depth.min()
    #depth_max = depth.max()

    #print (depth_min, depth_max)

    #max_val = (2**8)-1

    #if depth_max - depth_min > np.finfo("float").eps:
    #    #out = max_val * (depth - depth_min) / (depth_max - depth_min)
    #    out = depth * 0.1275
    #else:
    #    out = np.zeros(depth.shape, dtype=depth.type)

    cv2.imshow(WINDOW, out.astype("uint8"))
    #cv2.imshow("XZ", imgXZ)

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
