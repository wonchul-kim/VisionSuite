from abc import abstractmethod

class BaseTrainRunner:
    def __init__(self):
        pass 
    
    @abstractmethod
    def set_configs(self):
        pass 
    
    @abstractmethod
    def set_dataset(self):
        pass 
    
    def set_model(self):
        pass

    @abstractmethod
    def run_loop(self):
        pass 
    
    def train(self):
        self.set_configs()
        self.set_dataset()
        self.run_loop()
    