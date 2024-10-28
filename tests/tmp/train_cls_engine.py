from visionsuite.engines.classification.runners.train_runner import TrainRunner



if __name__ == "__main__":
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[2]
    
    runner = TrainRunner()
    runner.train(ROOT / "visionsuite/engines/classification/cfgs/datasets/rps.yaml")    

