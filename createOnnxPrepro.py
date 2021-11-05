import torch

# Model definition
class PreprocessModel(torch.nn.Module):
    def forward(self, x):
        # BRG to RGB
        x = x[:,:,:,[2, 1, 0]]
        #x = x.flip(3)
        # NHWC to NCHW, Byte to float, [0, 255] to [0, 1]
        x = x.permute(0,3,1,2).float() / 255.
        # Output [1,3,256,256] RGB float
        return x

# Input shape definition
batch_size = 1
height = 256
width = 256
channel = 3

# I/O names definition
input_names = ["inputByteArray"]
output_names = ['outputFloatArray']

# ONNX file name definition
onnx_file_name = "Preprocess{}x{}x{}xBGRxByte.onnx".format(batch_size, height, width)

# Export ONNX model
cpu = torch.device("cpu")
model = PreprocessModel().to(cpu)
x = torch.randn((batch_size, height, width, channel)).byte().to(cpu)
torch.onnx.export(model,
                x,
                onnx_file_name,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=input_names, 
                output_names=output_names
                )
print('Onnx model exporting done')
