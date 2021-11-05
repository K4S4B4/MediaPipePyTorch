import tflite2onnx

tflite_path = 'resource/MediaPipe/face_landmark_with_attention.tflite'
onnx_path = 'resource/MediaPipe/face_landmark_with_attention.onnx'

tflite2onnx.convert(tflite_path, onnx_path)