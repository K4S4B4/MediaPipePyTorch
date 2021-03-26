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

back_detector = True

#face_detector = BlazeFace(back_model=back_detector).to(gpu)
#if back_detector:
#    face_detector.load_weights("blazefaceback.pth")
#    face_detector.load_anchors("anchors_face_back.npy")
#else:
#    face_detector.load_weights("blazeface.pth")
#    face_detector.load_anchors("anchors_face.npy")

#palm_detector = BlazePalm().to(gpu)
#palm_detector.load_weights("blazepalm.pth")
#palm_detector.load_anchors("anchors_palm.npy")
#palm_detector.min_score_thresh = .75

hand_regressor = BlazeHandLandmark().to(gpu)
hand_regressor.load_weights("blazehand_landmark.pth")

#face_regressor = BlazeFaceLandmark().to(gpu)
#face_regressor.load_weights("blazeface_landmark.pth")


#WINDOW='test'
#cv2.namedWindow(WINDOW)
#if len(sys.argv) > 1:
#    capture = cv2.VideoCapture(sys.argv[1])
#    mirror_img = False
#else:
#    capture = cv2.VideoCapture(0)
#    mirror_img = True

#if capture.isOpened():
#    hasFrame, frame = capture.read()
#    frame_ct = 0
#else:
#    hasFrame = False

#img1, img2, scale, pad = resize_pad(frame)

#img1 = torch.tensor(img1, device=gpu).byte()
#img1 = img1.unsqueeze(0)
#normalized_landmarks2, flags2 = hand_regressor(img1)

##############################################################################
batch_size = 1
height = 256
width = 256
##############################################################################

##############################################################################
x = torch.randn((batch_size, height, width, 3), requires_grad=True).byte().to(gpu)
##############################################################################

#onnx_file_name = "BlazeHand_{}x{}x{}xBGRxByte_pose2dWithConf.onnx".format(batch_size, height, width)
input_names = ["input"] #[B,256,256,3],
output_names = ['joint3d', 'confidence'] #[B,21,3], [B]

onnx_file_name = "BlazeHand_b_{}_{}_BGRxByte.onnx".format(height, width)
dynamic_axes = {
    "input": {0: "batch_size"}, 
    "joint3d": {0: "batch_size"}, 
    "confidence": {0: "batch_size"}}

torch.onnx.export(hand_regressor,
                x,
                onnx_file_name,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names, 
                output_names=output_names,
                dynamic_axes=dynamic_axes)
print('Onnx model exporting done')
