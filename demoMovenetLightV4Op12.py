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

onnx_file_name = 'resource/MoveNet/movenet_singlepose_thunder_4_1x256x256x3xBGRxByte_opset12.onnx'
#onnx_file_name = 'resource/MoveNet/movenet_singlepose_lightning_4_1x192x192x3xBGRxByte_opset12.onnx'
input_size = 256
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_profiling = True
ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options)

input_name = ort_session.get_inputs()[0].name
    
while hasFrame:
    img1, img2, scale, pad = resize_pad(frame)

    img1 = cv2.resize(img1, (input_size,input_size))

    img_in = np.expand_dims(img1, axis=0).astype(np.uint8)
    ort_inputs = {input_name: img_in}

    ort_outs = ort_session.run(None, ort_inputs)

    imgDisp = img1.copy()

    landmarkTensor = ort_outs[0] #(1,1,17,3)
    landmark = landmarkTensor[0][0]

    for i in range(17):
        x, y = landmark[i][1], landmark[i][0]
        x, y = int(x * input_size), int(y * input_size)
        cv2.circle(img1, (x, y), 2, (0, 255, 0), thickness=2)


    img1 = cv2.resize(img1, (500,500))

    cv2.imshow(WINDOW, img1)
    #cv2.imshow("XZ", imgXZ)

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
