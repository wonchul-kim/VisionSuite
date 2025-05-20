import os
import json
import os.path as osp
from glob import glob 
import random 
from tqdm import tqdm 

ratio = 0.0
input_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/neuro/labelme'
output_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/neuro/neuro'

val_images = list(map(str, list(range(0, 51))))

os.makedirs(output_dir, exist_ok=True)

labelme_jsons = glob(osp.join(input_dir, "*.json"))

neuro_anns = {
    "label_type": 'seg',
    "source": "labelset",
    'classes': [],
    "data": [],
}

classes = []
for labelme_json in tqdm(labelme_jsons):
    json_file = osp.basename(labelme_json)
    filename = json_file.split('.')[0]
    img_file = filename + '.png'
    
    val = False
    if filename in val_images:
        val = True
    
    data = {
        'fileName': img_file, 
        "set": "train",
        "classLabel": "",
        "regionLabel": [],
        'retestset': 0,
        'rotation_angle': 0,
        'width': 1120,
        'height': 768
    }
    with open(labelme_json, 'r') as jf:
        labelme_anns = json.load(jf)['shapes']
        
    for labelme_ann in labelme_anns:
        label = labelme_ann['label']
        if label not in classes:
            classes.append(label)
        points = labelme_ann['points']
        
        data['regionLabel'].append({
            "className": label,
            "type": "PolyLine",
            "strokeWidth": 5,
            "points": [[int(x) for x in pair] for pair in points]
        })  
            
    neuro_anns['data'].append(data) 
    
    if val:
        data = {
            'fileName': str(int(filename) + 3835)+ '.png', 
            "set": "test",
            "classLabel": "",
            "regionLabel": [],
            'retestset': 0,
            'rotation_angle': 0,
            'width': 1120,
            'height': 768
        }
        with open(labelme_json, 'r') as jf:
            labelme_anns = json.load(jf)['shapes']
            
        for labelme_ann in labelme_anns:
            label = labelme_ann['label']
            if label not in classes:
                classes.append(label)
            points = labelme_ann['points']
            
            data['regionLabel'].append({
                "className": label,
                "type": "PolyLine",
                "strokeWidth": 5,
                "points": [[int(x) for x in pair] for pair in points]
            })  
                
        neuro_anns['data'].append(data)          
            
            
colors = ['rgba(248, 126, 172, 1)', 'rgba(167, 238, 62, 1)',
          'rgba(255, 180, 12, 1)', 'rgba(251, 92, 73, 1)']

for label, color in zip(classes, colors[:len(classes)]):
    neuro_anns['classes'].append({"name": label, "color": color})
                    
with open(osp.join(output_dir, 'neuro.json'), 'w') as json_file:
    json.dump(neuro_anns, json_file, ensure_ascii=False, indent=4)
    
    
# import os.path as osp 
# from glob import glob 
# from shutil import copyfile

# output_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/neuro/renamed_val_images'
# img_files = glob(osp.join('/HDD/datasets/projects/Tenneco/Metalbearing/outer/250211/neuro/val_iamges', '*.bmp'))

# for img_file in img_files:
#     filename = osp.split(osp.splitext(img_file)[0])[-1]
#     copyfile(img_file, osp.join(output_dir, str(int(filename) + 3835)+ '.bmp'))
    