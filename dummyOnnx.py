import cv2
import numpy as np
import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def forward(self, x):
        x = x[:,:,:,[2, 1, 0]] # BRG to RGB
        #x = x.permute(0,3,1,2).float() / 255.
        x = x.float() * 0.00392156862

        return x

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

model = DummyModel().to(gpu)

##############################################################################
batch_size = 1
height = 256
width = 256
x = torch.randn((batch_size, height, width, 3), requires_grad=True).byte().to(gpu)
##############################################################################

input_names = ['input_byte']
output_names = ['output_dummy']

onnx_file_name = "resource/Dummy_{}x{}x{}xBGRxByte.onnx".format(batch_size, height, width)
dynamic_axes = {
    "input_byte": {0: "batch_size"}, 
    "output_dummy": {0: "batch_size"}
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
