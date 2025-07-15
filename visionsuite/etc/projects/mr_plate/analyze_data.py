from glob import glob 
import os.path as osp 
import json 
from tqdm import tqdm 

input_dir = '/DeepLearning/research/data/unittests/seg_mr_benchmark'

folders = [folder.split("/")[-1] for folder in glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]
print(f"There are {len(folders)} folders")
print(f"     - {folders}")


info = {}
for folder in folders:
    json_files = glob(osp.join(input_dir, folder, '*.json'))
    
    if len(json_files) == 0:
        json_files = glob(osp.join(input_dir, folder, '*/*.json'))
    
    info[folder] = {'image_shape': set(), 'number of images': len(json_files)}
    for jdx, json_file in tqdm(enumerate(json_files), desc=folder):
        
        with open(json_file, 'r') as jf:
            anns = json.load(jf)
            
        width = anns['imageWidth']  
        height = anns['imageHeight']   
        info[folder]['image_shape'].add((width, height))
        
    print(info)
print(info)
    
import pandas as pd 

df = pd.DataFrame(data=info)
df.to_excel(osp.join(input_dir, 'info.xlsx'))

    
