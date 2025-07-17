from transformers import SegformerForSemanticSegmentation
from optimum.exporters.onnx import main_export
from pathlib import Path
import os.path as osp

output_dir = '/HDD/etc/outputs/onnx/segformer-b3'

model_id = "nvidia/segformer-b3-finetuned-ade-512-512"
output_dir = Path("onnx/segformer_b3")
main_export(
    model_name_or_path=model_id,
    output=output_dir,
    task="semantic-segmentation",
    opset=14
)

# import onnx
# import onnx_graphsurgeon as gs

# graph = gs.import_onnx(onnx.load(osp.join(output_dir, "model.onnx")))

# # 이름 변경
# graph.inputs[0].name = "data"
# graph.outputs[0].name = "output"

# onnx.save(gs.export_onnx(graph), osp.join(output_dir, "renamed.onnx"))