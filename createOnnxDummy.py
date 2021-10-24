import numpy as np
import torch
import cv2
import sys

class DummyModel(torch.nn.Module):
    def forward(self, x):
        x = x[:,:,:,[2, 1, 0]] # BRG to RGB
        x = x.permute(0,3,1,2).float() / 255. # NHWC to NCHW, Byte to float, [0, 255] to [0, 1]
        x2 = x * 2;
        x3 = x * 3;
        return x2, x3 

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

model = DummyModel().to(gpu)

##############################################################################
batch_size = 1
height = 256
width = 256
x = torch.randn((batch_size, height, width, 3), requires_grad=True).byte().to(gpu)
##############################################################################

input_names = ["inputName"] #[B,192,192,3],
output_names = ['xTimes2', 'xTimes3'] #[B,486,3], [B]

onnx_file_name = "Dummy_{}_{}_{}_xByte.onnx".format(batch_size, height, width)
dynamic_axes = {
    "inputName": {0: "batch_size"}, 
    "xTimes2": {0: "batch_size"}, 
    "xTimes3": {0: "batch_size"}
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
