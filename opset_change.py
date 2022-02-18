import onnx

model = onnx.load('resource/YOLOv4/yolov4tiny_1x416x416xBGRxByte.onnx')
op = onnx.OperatorSetIdProto()
op.version = 12
update_model = onnx.helper.make_model(model.graph, opset_imports=[op])
onnx.save(update_model, 'resource/YOLOv4/yolov4tiny_1x416x416xBGRxByte.opset12.onnx')
