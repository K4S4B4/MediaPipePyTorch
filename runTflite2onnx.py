import tflite2onnx

tflite_path = 'resource/MediaPipe/hand_recrop.tflite'
onnx_path = 'resource/MediaPipe/hand_recrop.onnx'

tflite2onnx.convert(tflite_path, onnx_path)