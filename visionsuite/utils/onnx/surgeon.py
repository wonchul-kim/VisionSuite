import onnx_graphsurgeon as gs
import onnx 



onnx_file = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/outputs/SEGMENTATION/segformer_b2_unfrozen/segformer_mit-b2_b2_w1120_h768.onnx'
graph = gs.import_onnx(onnx.load(onnx_file))

for idx, node in enumerate(graph.nodes):
    print(idx, '. ', node.op)
    if node.op == "Attention" or "MatMul" in node.op:
        # 단순 예시: 실제로는 subgraph pattern 매칭 필요
        plugin = gs.Node(
            op="MultiHeadAttentionPlugin",
            name="mha_plugin_%d" % id(node),
            inputs=node.inputs,
            outputs=node.outputs,
            attrs={"num_heads": 8, "embed_dim": 256},
        )
        graph.nodes.append(plugin)
        graph.nodes.remove(node)

onnx.save(gs.export_onnx(graph), "segformer_opt.onnx")
print("✅ segformer_opt.onnx plugin 적용 완료")