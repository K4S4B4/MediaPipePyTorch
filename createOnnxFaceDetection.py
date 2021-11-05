import numpy as np
import torch
import cv2
import sys

from blazeface import BlazeFace

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

back_detector = True

model = BlazeFace(back_model=back_detector).to(gpu)
if back_detector:
    model.load_weights("blazefaceback.pth")
    model.load_anchors("anchors_face_back.npy")
else:
    model.load_weights("blazeface.pth")
    model.load_anchors("anchors_face.npy")

##############################################################################
batch_size = 1
#height = 128
#width = 128
height = 256
width = 256
x = torch.randn((batch_size, height, width, 3), requires_grad=True).byte().to(gpu)
opset = 12
##############################################################################

input_names = ["input"] #[B,192,192,3],
output_names = ['detection', 'confidence'] #[B,486,3], [B]

onnx_file_name = "BlazeFaceDetection_{}x{}x{}xBGRxByte_opset{}.onnx".format(batch_size, height, width, opset)
dynamic_axes = {
    "input": {0: "batch_size"}, 
    "detection": {0: "batch_size"}, 
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
