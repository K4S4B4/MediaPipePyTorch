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

model = BlazeFaceLandmark().to(gpu)
model.load_weights("blazeface_landmark.pth")

##############################################################################
batch_size = 1
height = 192
width = 192
x = torch.randn((batch_size, height, width, 3), requires_grad=True).byte().to(gpu)
opset = 12
##############################################################################

input_names = ["input"] #[B,192,192,3],
output_names = ['landmark', 'confidence'] #[B,486,3], [B]

onnx_file_name = "BlazeFace_{}_{}_{}_BGRxByte_opset{}.onnx".format(batch_size, height, width, opset)
dynamic_axes = {
    "input": {0: "batch_size"}, 
    "landmark": {0: "batch_size"}, 
    "confidence": {0: "batch_size"}
    }

torch.onnx.export(model,
                x,
                onnx_file_name,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=input_names, 
                output_names=output_names
                #,dynamic_axes=dynamic_axes
                )
print('Onnx model exporting done')
