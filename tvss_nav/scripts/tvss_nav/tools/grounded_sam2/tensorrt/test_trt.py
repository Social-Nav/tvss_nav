import tensorrt as trt

print("TRT Version:", trt.__version__)

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

with open("trt/hiera_t_image_encoder.trt", "rb") as f:
    engine_data = f.read()

engine = runtime.deserialize_cuda_engine(engine_data)

print("Engine:", engine)
assert engine is not None, "[Failed] Engine deserialization failed!"
print("[Succese] TensorRT engine loaded successfully.")