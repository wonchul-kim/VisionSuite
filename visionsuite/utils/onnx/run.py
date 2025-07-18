import torch
import onnxruntime as ort
import numpy as np
import os.path as osp
import json
import time 

_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/outputs/SEGMENTATION/segformer_b2_unfrozen/weights'
onnx_filename = 'segformer_mit-b2_b1_w1120_h768_custom'
# onnx_filename = 'segformer_mit-b2_b1_w1120_h768'
opts = ort.SessionOptions()
opts.enable_profiling = True


session = ort.InferenceSession(osp.join(_dir, onnx_filename) + '.onnx',
                               sess_options=opts,
                               providers=['CUDAExecutionProvider'])

input_name = session.get_inputs()[0].name
dummy_input = np.random.rand(1, 3, 768, 1120).astype(np.float32)  # 예시
st = time.time()
output = session.run(None, {input_name: dummy_input})
print("TIME: ", time.time() - st)
# 프로파일 결과 파일명 추출 및 분석
profile_file = session.end_profiling()
print(f"profile_file: {profile_file}")

if not profile_file or not osp.isfile(profile_file):
    print("❌ 프로파일 파일이 제대로 생성되지 않았습니다.")
else:
    print("✅ 프로파일 파일이 성공적으로 생성되었습니다!")
    
with open(profile_file, 'r') as f:
    profile = json.load(f)

# 시간 많이 걸린 연산 TOP 10 추출 및 출력
timed_ops = []
for d in profile:
    if 'name' in d:
        if 'op_name' not in d['args']:
            timed_ops.append((d['name'], d['dur'], ''))
        else:
            timed_ops.append((d['name'], d['dur'], d['args']['op_name']))


# timed_ops = [(d['args']['name'], d['dur']) for d in profile if 'name' in d['args']]
timed_ops.sort(key=lambda x: x[1], reverse=True)
print("실행 시간 상위 연산:")
for name, duration, op_name in timed_ops[:30]:
    print(f"{name} >>> {op_name}: {duration/1e6} ms")
