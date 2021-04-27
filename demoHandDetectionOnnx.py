import numpy as np
import cv2
import sys
from blazebase import resize_pad, denormalize_detections
from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS
import onnxruntime
import math


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

onnx_file_name = 'resource/MediaPipe/hand_recrop.onnx'
sess_options = onnxruntime.SessionOptions()
#sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
#sess_options.enable_profiling = True
ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options)

input_name = ort_session.get_inputs()[0].name
    
while hasFrame:
    img1, img2, scale, pad = resize_pad(frame)

    img_in = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in /= 255.0
    img_in = np.expand_dims(img_in, axis=0)

    ort_inputs = {input_name: img_in}

    ort_outs = ort_session.run(None, ort_inputs)

    x = ort_outs[0][0][0][0][0]
    y = ort_outs[0][0][1][0][0]
    u = ort_outs[0][0][2][0][0]
    v = ort_outs[0][0][3][0][0]

    #y = (ort_outs[0][0][1][0][0] + ort_outs[0][0][3][0][0]) / 2
    #x = (ort_outs[0][0][0][0][0] + ort_outs[0][0][2][0][0]) / 2
    #scale = (ort_outs[0][0][3][0][0] - ort_outs[0][0][1][0][0])

    ##y += -0.5 * scale
    ##scale *= 2.6
    #w = scale

    width = math.sqrt( (x-u)*(x-u) + (y-v)*(y-v) )

    cv2.circle(img1, (int(x), int(y)), int(width), (255,0,0))
    cv2.circle(img1, (int(x), int(y)), 10, (0,255,0))
    cv2.circle(img1, (int(u), int(v)), 10, (0,255,255))
    #cv2.circle(img1, (int(x + w * np.cos(t / 180 * np.pi)), int(y + w * np.sin(t/ 180 * np.pi))), 10, (0,255,0))

    #palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

    #for i in range(len(ort_outs[0])):
    #    landmark, flag, hanndedness = ort_outs[0][i], ort_outs[1][i], ort_outs[2][i]
    #    if flag > 0.2:
    #        if hanndedness > 0.9 or hanndedness < 0.1:
    #            draw_landmarks(img1, landmark[:,:2], HAND_CONNECTIONS, size=2)
    #            print(hanndedness)

    cv2.imshow(WINDOW, img1)

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
