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



#pose_detector = BlazePose().to(gpu)
#pose_detector.load_weights("blazepose.pth")
#pose_detector.load_anchors("anchors_pose.npy")

#pose_regressor = BlazePoseLandmark().to(gpu)
#pose_regressor.load_weights("blazepose_landmark.pth")

onnx_file_name = 'BlazePoseDetection_1x128x128xBGRxByte.onnx'
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_profiling = True
ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options)
input_name = ort_session.get_inputs()[0].name

#onnx_file_name2 = 'resource\MediaPipe\pose_landmark_lite_1x256x256x3xBGRxByte.onnx'
onnx_file_name2 = 'resource\MediaPipe\pose_landmark_full_1x256x256x3xBGRxByte.onnx'
#onnx_file_name2 = 'resource\MediaPipe\pose_landmark_heavy_1x256x256x3xBGRxByte.onnx'
sess_options2 = onnxruntime.SessionOptions()
sess_options2.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options2.enable_profiling = True
ort_session2 = onnxruntime.InferenceSession(onnx_file_name2, sess_options2)
input_name2 = ort_session2.get_inputs()[0].name


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

    #frame = np.ascontiguousarray(frame[:,::-1,::-1])

    img1, img2, scale, pad = resize_pad(frame)

    ##normalized_pose_detections = pose_detector.predict_on_image(img2)
    #img = torch.from_numpy(img2).unsqueeze(0).to(gpu)
    #normalized_pose_detections = pose_detector(img)

    img_in = np.expand_dims(img2, axis=0).astype(np.uint8)
    ort_inputs = {input_name: img_in}
    ort_outs = ort_session.run(None, ort_inputs)
    normalized_pose_detections = torch.from_numpy(ort_outs[0][0]).to(gpu)

    #print(ort_outs[1])

    #img_in2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    #img_in2 = img1
    #img_in2 = np.expand_dims(img_in2, axis=0).astype(np.uint8) / 255

    img_in2 = np.expand_dims(img1, axis=0).astype(np.uint8)
    ort_inputs2 = {input_name2: img_in2}
    ort_outs2 = ort_session2.run(None, ort_inputs2)

    #print(ort_outs2[1])


    if ort_outs2[1] > 0.5:
        pose_detections = denormalize_detections(normalized_pose_detections, scale, pad)

        #cv2.circle(frame, (int(pose_detections[0, 0]), int(pose_detections[0, 1])), 2, (255, 255, 0), thickness=2)
        #cv2.circle(frame, (int(pose_detections[0, 2]), int(pose_detections[0, 3])), 2, (0, 0, 0), thickness=2)
        cv2.circle(frame, (int(pose_detections[0, 4]), int(pose_detections[0, 5])), 2, (255, 255, 255), thickness=2)
        #cv2.circle(frame, (int(pose_detections[0, 6]), int(pose_detections[0, 7])), 2, (255, 0, 0), thickness=2)
        #cv2.circle(frame, (int(pose_detections[0, 8]), int(pose_detections[0, 9])), 2, (0, 255, 0), thickness=2)
        #cv2.circle(frame, (int(pose_detections[0, 10]),int(pose_detections[0, 11])), 2, (0, 0, 255), thickness=2)

        for n in range(0, 33):
            cv2.circle(frame, (int(ort_outs2[0][0, 5*n + 0]* scale)- pad[1],int(ort_outs2[0][0, 5*n + 1]* scale) - pad[0]), 5, (0, 0, 0), thickness=2)

        #for n in range(33, 39):
            #cv2.circle(frame, (int(ort_outs2[0][0, 5*n + 0]* scale)- pad[1],int(ort_outs2[0][0, 5*n + 1]* scale)- pad[0]), 5, (255, 255, 255), thickness=2)

            #cv2.circle(frame, (int(ort_outs2[4][0, 3*n + 0]) * 256 + 128,int(ort_outs2[4][0, 3*n + 1]))* 256 + 128, 2, (255, 0, 0), thickness=2)
            #print(ort_outs2[0][0, 5*n + 0] - ort_outs2[4][0, 3*n + 0]* 256 - 128)

        n = 33
        cv2.circle(frame, (int(ort_outs2[0][0, 5*n + 0]* scale)- pad[1],int(ort_outs2[0][0, 5*n + 1]* scale)- pad[0]), 5, (255, 255, 255), thickness=2)
        n = 34
        cv2.circle(frame, (int(ort_outs2[0][0, 5*n + 0]* scale)- pad[1],int(ort_outs2[0][0, 5*n + 1]* scale)- pad[0]), 5, (255, 0, 0), thickness=2)
        n = 35
        cv2.circle(frame, (int(ort_outs2[0][0, 5*n + 0]* scale)- pad[1],int(ort_outs2[0][0, 5*n + 1]* scale)- pad[0]), 5, (0, 255, 0), thickness=2)
        n = 36
        cv2.circle(frame, (int(ort_outs2[0][0, 5*n + 0]* scale)- pad[1],int(ort_outs2[0][0, 5*n + 1]* scale)- pad[0]), 5, (0, 0, 255), thickness=2)
        n = 37
        cv2.circle(frame, (int(ort_outs2[0][0, 5*n + 0]* scale)- pad[1],int(ort_outs2[0][0, 5*n + 1]* scale)- pad[0]), 5, (255, 127, 127), thickness=2)
        n = 38
        cv2.circle(frame, (int(ort_outs2[0][0, 5*n + 0]* scale)- pad[1],int(ort_outs2[0][0, 5*n + 1]* scale)- pad[0]), 5, (127, 255, 127), thickness=2)

        #print(  ort_outs2[0][0, 5*0 + 3]
        #        , ort_outs2[0][0, 5*1 + 3]
        #        , ort_outs2[0][0, 5*2 + 3]
        #        , ort_outs2[0][0, 5*3 + 3]
        #        , ort_outs2[0][0, 5*4 + 3]
        #        , ort_outs2[0][0, 5*5 + 3]
        #        , ort_outs2[0][0, 5*6 + 3]
        #        , ort_outs2[0][0, 5*7 + 3]
        #        , ort_outs2[0][0, 5*8 + 3]
        #        , ort_outs2[0][0, 5*9 + 3]
        #        , ort_outs2[0][0, 5*10 + 3]
        #        , ort_outs2[0][0, 5*11 + 3]
        #        , ort_outs2[0][0, 5*12 + 3]
        #        , ort_outs2[0][0, 5*13 + 3]
        #        , ort_outs2[0][0, 5*14 + 3]
        #        , ort_outs2[0][0, 5*15 + 3]
        #        , ort_outs2[0][0, 5*16 + 3]
        #        , ort_outs2[0][0, 5*17 + 3]
        #        , ort_outs2[0][0, 5*18 + 3]
        #        , ort_outs2[0][0, 5*19 + 3]
        #        , ort_outs2[0][0, 5*20 + 3]
        #        , ort_outs2[0][0, 5*21 + 3]
        #        , ort_outs2[0][0, 5*22 + 3]
        #        , ort_outs2[0][0, 5*23 + 3]
        #        , ort_outs2[0][0, 5*24 + 3]
        #        , ort_outs2[0][0, 5*25 + 3]
        #        , ort_outs2[0][0, 5*26 + 3]
        #        , ort_outs2[0][0, 5*27 + 3]
        #        , ort_outs2[0][0, 5*28 + 3]
        #        , ort_outs2[0][0, 5*29 + 3]
        #        , ort_outs2[0][0, 5*30 + 3]
        #        , ort_outs2[0][0, 5*31 + 3]
        #        , ort_outs2[0][0, 5*32 + 3]
        #        , ort_outs2[0][0, 5*33 + 3]
        #        , ort_outs2[0][0, 5*34 + 3]
        #        , ort_outs2[0][0, 5*35 + 3]
        #        , ort_outs2[0][0, 5*36 + 3]
        #        , ort_outs2[0][0, 5*37 + 3]
        #        , ort_outs2[0][0, 5*38 + 3]
        #        )

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

    frame = cv2.resize(frame, dsize=None, fx = 2, fy =2)
    cv2.imshow(WINDOW, frame)
    # cv2.imwrite('sample/%04d.jpg'%frame_ct, frame[:,:,::-1])

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
