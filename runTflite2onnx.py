import tflite2onnx

tflite_path = 'resource/MediaPipe/pose_detection.tflite'
onnx_path = 'resource/MediaPipe/pose_detection_original.onnx'

tflite2onnx.convert(tflite_path, onnx_path)