from typing import Union 

from visionsuite.cores.roboflow.utils import labelme2yolo_iseg, labelme2yolo_hbb

class DatasetConverter:
    
    def __init__(self, task):
        self.__task = task

        self._set_converter()
        
    @property 
    def task(self):
        return self.__task 
    
    @task.setter
    def task(self, value):
        self.__task = value
    
    def _set_converter(self):
        if self.__task == 'hbb_detection':
            self._converter = labelme2yolo_hbb
        elif self.__task == 'instance_segmentation':
            self._converter = labelme2yolo_iseg
        elif self.__task == 'obb_detection':
            pass
        else:
            raise NotImplementedError(f"There is no such converter for {self.__task}")
        
    def run(self, input_dir: str, output_dir: str, 
                        copy_image: bool, image_ext: Union[list, str]):
        
        self._converter(input_dir, output_dir, copy_image, image_ext)
            
        


if __name__ == '__main__':
    input_dir = '/HDD/datasets/projects/visionsuite/yolo/hbb_detection/split_dataset'
    output_dir = '/HDD/datasets/projects/visionsuite/yolo/hbb_detection/split_dataset_yolo_hbb'

    copy_image = True
    image_ext = 'png'
       
    converter = DatasetConverter('hbb_detection')
    converter.run(input_dir, output_dir, copy_image, image_ext)
    
            