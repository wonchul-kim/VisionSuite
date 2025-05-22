import os.path as osp
import os 
import shutil 
from pathlib import Path
from tqdm import tqdm
from glob import glob 


def make_repeatability_test_dataset(input_dir, output_dir, filenames_dir, cases, exception_txt):
    
    exception_list = []
    if osp.exists(exception_txt):
        f = open(exception_txt, 'r')
        while True:
            line= f.readline()
            if not line: break 
            
            exception_list.append(line.lstrip().rstrip())
            
        f.close()
    print(f'Exception: {exception_list}')
    
    
    
    categories = os.listdir(filenames_dir)
    print(f"Categories: {categories}")
    cat2list = {}
    for category in categories:
        if category in ['vis', '.DS_Store']:
            continue
        _filenames_dir = osp.join(filenames_dir, category)
        filenames = glob(osp.join(_filenames_dir, "*.bmp"))
        _filenames = []
        for filename in filenames:
            filename = osp.split(osp.splitext(filename)[0])[-1]
            _filenames.append(filename)
            
        cat2list[category] = _filenames
        
    for case in cases:
        base_dir = osp.join(input_dir, str(case))
        folders = sorted([Path(base_dir) / f for f in os.listdir(base_dir) if (Path(base_dir) / f).is_dir()])
        # sorted_folders = sorted(folders, key=lambda f: f.stat().st_ctime)
        print(f"There are {len(folders)} folders")
        
        if 'OUTER_shot0' in case:
            _case = case.replace('OUTER_shot0', '')
        else:
            _case = case
        
        for idx, folder_dir in tqdm(enumerate(folders), desc=f"case: {str(case)}"):
            folder_dir = str(folder_dir)
            filename = folder_dir.split('/')[-1]
            
            if '_Outer' in filename:
                filename = filename.replace('_Outer', '')
            
            found = False
            _filename = filename.split("_")[0]
            for cat, image_list in cat2list.items():
                    
                if _filename in image_list:
                    
                    _output_dir = osp.join(output_dir, cat, str(_case), filename)
            
                    if not osp.exists(_output_dir):
                        os.makedirs(_output_dir)

                    img_file = osp.join(folder_dir, f"1_image.bmp")
                    assert osp.exists(img_file), ValueError(f"There is no such image file: {img_file}")
                    
                    shutil.copyfile(img_file, osp.join(_output_dir, '1_image.bmp'))
                    
                    found = True

            if not found:
                if _filename in exception_list:
                    continue
                
                _output_dir = osp.join(output_dir, 'ok', str(_case), filename)
        
                if not osp.exists(_output_dir):
                    os.makedirs(_output_dir)
                
                img_file = osp.join(folder_dir, f"1_image.bmp")
                assert osp.exists(img_file), ValueError(f"There is no such image file: {img_file}")
                
                shutil.copyfile(img_file, osp.join(_output_dir, '1_image.bmp'))

if __name__ == '__main__':                
    # ## 1st
    # input_dir = '/Data/01.Image/research/benchmarks/production/tenneco/repeatibility/v01/final_data'
    # filenames_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeat_250515/1st/outputs/not_repeated/'
    # exception_txt = osp.join(filenames_dir, '../../exception.txt')
    # output_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeat_250515/1st/data'
    # cases = ['OUTER_shot01', 'OUTER_shot02', 'OUTER_shot03']
    
    ### 2nd
    input_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer_repeatability/2nd/data'
    filenames_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeat_250515/2nd/outputs/not_repeated/'
    exception_txt = osp.join(filenames_dir, '../../exception.txt')
    output_dir = '/Data/01.Image/Tenneco/Metalbearing/4_FOR_SIMULATION/TEST_SET/repeat_250515/2nd/data'
    cases = ['1', '2', '3']
    
    make_repeatability_test_dataset(input_dir, output_dir, filenames_dir, cases, exception_txt)