from bases import BaseEngine

class Engine(BaseEngine):
    def __init__(self, mode):
        super().__init__('classificatoin')

        self._mode = mode 