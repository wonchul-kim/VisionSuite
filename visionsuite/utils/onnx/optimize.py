import onnx
import onnxoptimizer

model = onnx.load("/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/outputs/SEGMENTATION/segformer_b2_unfrozen/segformer_mit-b2_b2_w1120_h768.onnx")
passes = [
    "eliminate_nop_transpose",
    "fuse_transpose_into_gemm",
    "eliminate_identity",
    "eliminate_unused_initializer",
    "eliminate_nop_pad",
]
optimized_model = onnxoptimizer.optimize(model, passes)
onnx.save(optimized_model, "/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/outputs/SEGMENTATION/segformer_b2_unfrozen/segformer_mit-b2_b2_w1120_h768_optimized.onnx")
