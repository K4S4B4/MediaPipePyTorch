import numpy as np
import torch
import cv2
import onnxruntime

from blazebase import resize_pad, denormalize_detections
from blazepose import BlazePose
from blazepose_landmark import BlazePoseLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, POSE_CONNECTIONS

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)



pose_detector = BlazePose().to(gpu)
pose_detector.load_weights("blazepose.pth")
pose_detector.load_anchors("anchors_pose.npy")

pose_regressor = BlazePoseLandmark().to(gpu)
pose_regressor.load_weights("blazepose_landmark.pth")

onnx_file_name = 'BlazePoseDetection_1x128x128xBGRxByte.onnx'
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_profiling = True
ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options)
input_name = ort_session.get_inputs()[0].name

WINDOW='test'
cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(2)

if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False

while hasFrame:
    frame_ct +=1

    frame = np.ascontiguousarray(frame[:,::-1,::-1])

    img1, img2, scale, pad = resize_pad(frame)

    ##normalized_pose_detections = pose_detector.predict_on_image(img2)
    #img = torch.from_numpy(img2).unsqueeze(0).to(gpu)
    #normalized_pose_detections = pose_detector(img)

    img_in = np.expand_dims(img2, axis=0).astype(np.uint8)
    ort_inputs = {input_name: img_in}
    ort_outs = ort_session.run(None, ort_inputs)
    normalized_pose_detections = torch.from_numpy(ort_outs[0][0]).to(gpu)

    print(ort_outs[1])

    if ort_outs[1][0,0,0] > 0.5:
        pose_detections = denormalize_detections(normalized_pose_detections, scale, pad)

        #cv2.circle(frame, (int(pose_detections[0, 0]), int(pose_detections[0, 1])), 2, (255, 255, 0), thickness=2)
        #cv2.circle(frame, (int(pose_detections[0, 2]), int(pose_detections[0, 3])), 2, (0, 0, 0), thickness=2)
        cv2.circle(frame, (int(pose_detections[0, 4]), int(pose_detections[0, 5])), 2, (255, 255, 255), thickness=2)
        #cv2.circle(frame, (int(pose_detections[0, 6]), int(pose_detections[0, 7])), 2, (255, 0, 0), thickness=2)
        #cv2.circle(frame, (int(pose_detections[0, 8]), int(pose_detections[0, 9])), 2, (0, 255, 0), thickness=2)
        cv2.circle(frame, (int(pose_detections[0, 10]),int(pose_detections[0, 11])), 2, (0, 0, 255), thickness=2)


        #xc, yc, scale, theta = pose_detector.detection2roi(pose_detections)
        #img, affine, box = pose_regressor.extract_roi(frame, xc, yc, theta, scale)
        #flags, normalized_landmarks, mask = pose_regressor(img.to(gpu))
        #landmarks = pose_regressor.denormalize_landmarks(normalized_landmarks, affine)

        #draw_detections(frame, pose_detections)
        #draw_roi(frame, box)

        #for i in range(len(flags)):
        #    landmark, flag = landmarks[i], flags[i]
        #    if flag>.5:
        #        draw_landmarks(frame, landmark, POSE_CONNECTIONS, size=2)

    cv2.imshow(WINDOW, frame[:,:,::-1])
    # cv2.imwrite('sample/%04d.jpg'%frame_ct, frame[:,:,::-1])

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
