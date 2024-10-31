

class TrainParams:
    """
        Organize parameters to be used for training
    """
    def __init__(self):
        self._current_epoch = 0
        
        
    @property
    def current_epoch(self):
        return self._current_epoch
    
    @current_epoch.setter
    def current_epoch(self, val):
        self._current_epoch = val
    