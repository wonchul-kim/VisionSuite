from visionsuite.engines.segmentation import Engine


if __name__ == "__main__":
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[2]
    
    runner = Engine('segmentation')
    runner.train(ROOT / "visionsuite/engines/segmentation/cfgs/datasets/tenneco_inner.yaml")  
    # runner.train(ROOT / "visionsuite/engines/segmentation/cfgs/datasets/sungwoo_bottom.yaml")    
    # runner.train(ROOT / "visionsuite/engines/segmentation/cfgs/datasets/coco.yaml")    
