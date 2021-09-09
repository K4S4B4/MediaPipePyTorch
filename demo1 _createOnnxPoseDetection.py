import numpy as np
import torch
import cv2
import sys

from blazebase import resize_pad, denormalize_detections
from blazepose import BlazePose
from blazepose_landmark import BlazePoseLandmark
from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

model = BlazePose().to(gpu)
model.load_weights("blazepose.pth")
model.load_anchors("anchors_pose.npy")

##############################################################################
batch_size = 1
height = 128
width = 128
x = torch.randn((batch_size, height, width, 3), requires_grad=True).byte().to(gpu)
##############################################################################

input_names = ["input"] #[B,192,192,3],
output_names = ['detection'] #[B,486,3], [B]

onnx_file_name = "BlazePoseDetection_1x128x128xBGRxByte.onnx".format(batch_size, height, width)
dynamic_axes = {
    "input": {0: "batch_size"}, 
    "detection": {0: "batch_size"}
    }

torch.onnx.export(model,
                x,
                onnx_file_name,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names, 
                output_names=output_names
                #,dynamic_axes=dynamic_axes
                )
print('Onnx model exporting done')
