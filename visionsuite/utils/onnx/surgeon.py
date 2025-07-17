import onnx_graphsurgeon as gs
import onnx 
import os.path as osp

_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/outputs/SEGMENTATION/segformer_b2_unfrozen/weights'
onnx_filename = 'segformer_mit-b2_b1_w1120_h768'
graph = gs.import_onnx(onnx.load(osp.join(_dir, onnx_filename) + '.onnx'))

new_nodes = []
nodes_to_remove = []
for idx, node in enumerate(graph.nodes):
    
    if node.name == '/backbone/layers.0.1.0/ffn/Transpose':
        next_nodes = [n for n in graph.nodes if n.inputs and n.inputs[0] == node.outputs[0]]
        if len(next_nodes) == 1 and next_nodes[0].op == "Reshape":
            reshape_node = next_nodes[0]

            # Custom node 생성 (MyNlcToNchw)
            custom_node = gs.Node(
                op="MyNlcToNchw",
                name="Fused_NlcToNchw",
                inputs=node.inputs,
                outputs=reshape_node.outputs
            )
            new_nodes.append(custom_node)
            nodes_to_remove.extend([node, reshape_node])

for node in nodes_to_remove:
    graph.nodes.remove(node)

graph.nodes.extend(new_nodes)
onnx.save(gs.export_onnx(graph), osp.join(_dir, onnx_filename) + "_custom.onnx")