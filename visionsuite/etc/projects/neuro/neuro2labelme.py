import os 
import os.path as osp 
from glob import glob 
import json 
from visionsuite.utils.dataset.formats.labelme.utils import init_labelme_json, add_labelme_element

orders = ['1', '2', '3']

input_dir = '/DeepLearning/etc/neuro/1st/경계성'
output_dir = '/DeepLearning/etc/neuro/1st/경계성/labelme'

os.makedirs(output_dir, exist_ok=True)


for order in orders:

    json_file = osp.join(input_dir, f'{order}.json')
    with open(json_file, 'r') as jf:
        anns = json.load(jf)
    w, h = 1120, 768
        
    for data in anns['data']:
        
        filename = data['fileName'].split('.')[0]
        region_label = data['regionLabel']
        _labelme = init_labelme_json(filename + '.bmp', w, h)
        
        for label in region_label:
            _labelme = add_labelme_element(_labelme, shape_type='polygon', 
                                        label=label['className'], 
                                        points=label['points'])


        _output_dir = osp.join(output_dir, order)

        os.makedirs(_output_dir, exist_ok=True)

        with open(os.path.join(_output_dir, filename + ".json"), "w") as jsf:
            json.dump(_labelme, jsf)
        
        
            
