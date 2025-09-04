# Export SAM models as TensorRT models



```bash
python export_to_onnx.py

trtexec --onnx=tensorrt/onnx/hiera_t_image_encoder.onnx --saveEngine=tensorrt/trt/hiera_t_image_encoder.trt --inputIOFormats=fp32:chw --outputIOFormats=fp16:chw --fp16
```
