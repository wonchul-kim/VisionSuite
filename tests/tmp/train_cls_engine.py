from visionsuite.engines.classification import Engine


if __name__ == "__main__":
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[2]
    
    runner = Engine('classification')
    runner.train(ROOT / "visionsuite/engines/classification/cfgs/datasets/rps.yaml")    
    # runner.train(ROOT / "visionsuite/engines/classification/cfgs/datasets/cifar10.yaml")    

