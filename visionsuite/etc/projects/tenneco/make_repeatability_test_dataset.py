import os.path as osp
import os 
import shutil 
from pathlib import Path
from tqdm import tqdm



def make_repeatability_test_dataset(input_dir, output_dir):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    cases = [1, 2, 3]
    fovs = list(range(1, 15))
    for case in cases:
        base_dir = osp.join(input_dir, str(case))
        folders = sorted([Path(base_dir) / f for f in os.listdir(base_dir) if (Path(base_dir) / f).is_dir()])
        # sorted_folders = sorted(folders, key=lambda f: f.stat().st_ctime)
        print(f"There are {len(folders)} folders")

        _output_dir = osp.join(output_dir, str(case))
        
        if case == 1:
            sorted_folders_case_1 = folders
        
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)

        for idx, folder_dir in tqdm(enumerate(folders), desc=f"case: {str(case)}"):
            folder_dir = str(folder_dir)
            for fov in fovs:
                img_file = osp.join(folder_dir, f"{fov}_Outer.bmp")
                assert osp.exists(img_file), ValueError(f"There is no such image file: {img_file}")
                
                __output_dir = osp.join(_output_dir, str(sorted_folders_case_1[idx]).split("/")[-1] + f'_{fov}')
                
                if not osp.exists(__output_dir):
                    os.mkdir(__output_dir)
                
                shutil.copyfile(img_file, osp.join(__output_dir, '1_image.bmp'))
                


                
                
                
if __name__ == '__main__':                
    input_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/benchmark/반복성/250328/PC2'
    output_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/repeatability'
    make_repeatability_test_dataset(input_dir, output_dir)