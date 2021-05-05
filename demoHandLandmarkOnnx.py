import numpy as np
import torch
import cv2
import sys
from blazebase import resize_pad, denormalize_detections
from visualization import draw_detections,  draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS
import onnxruntime


def draw_landmarks(img, points, shiftY, connections=[], color=(0, 255, 0), size=2):
    points = points[:,:2]
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
    capture = cv2.VideoCapture(0)
    mirror_img = True

if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False

onnx_file_name = 'resource/MediaPipe/BlazeHand_B_256_256_BGRxByte.onnx'
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

    imgDisp = img1.copy()

    for i in range(len(ort_outs[0])):
        landmark, flag, hanndedness = ort_outs[0][i], ort_outs[1][i], ort_outs[2][i]
        if flag > 0.2:
            if hanndedness > 0.9 or hanndedness < 0.1:
                draw_landmarks(imgDisp, landmark[:,:2], 0, HAND_CONNECTIONS, size=2)
                print(hanndedness)

    imgXZ = img1.copy()
    landmarkXZ = landmark[:,[0,2,1]]
    if flag > 0.2:
        if hanndedness > 0.9 or hanndedness < 0.1:
            draw_landmarks(imgXZ, landmarkXZ[:,:2], 128, HAND_CONNECTIONS, size=2)
            print(hanndedness)


    cv2.imshow(WINDOW, imgDisp)
    cv2.imshow("XZ", imgXZ)

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
