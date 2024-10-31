

from typing import Any

def auto_initialize(init_func):
    def wrapper(self, *args, **kwargs):
        object.__setattr__(self, '_initialized', False)  # 초기화 중임을 표시
        result = init_func(self, *args, **kwargs)  # 원래의 __init__ 호출
        object.__setattr__(self, '_initialized', True)  # 초기화 완료 표시
        return result
    return wrapper

class TrainParams:
    """
        Organize parameters to be used for training
    """
    @auto_initialize
    def __init__(self):
        self._current_epoch = 0
        
    def __setattr__(self, name: str, value: Any) -> None:
        if not hasattr(self, '_initialized') or not self._initialized or name in self.__dict__:
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"Cannot add new attribute: {name}")
        
    @property
    def current_epoch(self):
        return self._current_epoch
    
    @current_epoch.setter
    def current_epoch(self, val):
        self._current_epoch = val
    
    
if __name__ == '__main__':
    params = TrainParams()
    print(params.current_epoch)
    params.a = 0
    print(params.a)