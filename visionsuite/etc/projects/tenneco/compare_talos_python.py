from glob import glob 
import os.path as osp 
from tqdm import tqdm


talos_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeatability/2nd/talos result/same_ok'
python_txt_file = '/HDD/etc/repeatablility/2nd/deeplabv3plus_2/filenames/IoU-0.05_Area-20_Conf-0_no_diff_no_points.txt'

talos_img_files = glob(osp.join(talos_dir, '*.jpg'))

talos_ids = []
for talos_img_file in tqdm(talos_img_files):
    filename = osp.split(osp.splitext(talos_img_file)[0])[-1]
    
    talos_ids.append(filename)
        
python_ids = []
python_txt = open(python_txt_file, 'r')
# while True: 
#     line = python_txt.readline()
#     if line is None: break
    
#     python_ids.append(line.replace('\n', ''))
    
with open(python_txt_file, "r", encoding="utf-8") as file:
    python_ids = [line.strip() for line in file]

talos_ids = set(talos_ids)
python_ids = set(python_ids)

print("talos: ", len(talos_ids))
print("python: ", len(python_ids))
print("Different from talos: ", len(talos_ids.difference(python_ids)))
print("Different from python: ", len(python_ids.difference(talos_ids)))
print("Same: ", len(talos_ids.intersection(python_ids)))