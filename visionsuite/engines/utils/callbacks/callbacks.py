from .default import get_default_callbacks

class Callbacks:

    def add_integration_callbacks(cls, engine):
        callbacks_list = []

        # if "Train" in engine.__class__.__name__:
        #     if engine._task.lower() == "segmentation":
        #         from .tools.segmentation_cb import callbacks as seg_cb

        #         callbacks_list.extend([seg_cb])
        #     elif engine._task.lower() == "detection":
        #         from .tools.detection_cb import callbacks as det_cb

        #         callbacks_list.extend([det_cb])
        #     elif engine._task.lower() == "obbdetection":
        #         from .tools.obbdetection_cb import callbacks as obb_det_cb

        #         callbacks_list.extend([obb_det_cb])
        #     elif engine._task.lower() == "classification":
        #         from .tools.classification_cb import callbacks as cls_cb

        #         callbacks_list.extend([cls_cb])
        #     elif engine._task.lower() == "ocr":
        #         from aivocr.utils.callbacks.train_cb import callbacks as ocr_cb

        #         callbacks_list.extend([ocr_cb])
        #     else:
        #         raise NotImplementedError(f"There is no such task: {engine._task}")

        #     if cls.logger is not None:
        #         cls.logger.info(
        #             f"Added {engine._task} callbacks are integrated",
        #             cls.add_integration_callbacks.__name__,
        #             __class__.__name__,
        #         )

        # if engine.configs.logging.wb:
        #     from .tools.wandb_cb import callbacks as wb_cb

        #     callbacks_list.extend([wb_cb])
        #     if cls.logger is not None:
        #         cls.logger.info(
        #             f"Added Wandb callbacks are integrated",
        #             cls.add_integration_callbacks.__name__,
        #             __class__.__name__,
        #         )
 
        for callbacks in callbacks_list:
            for k, v in callbacks.items():
                if v not in cls._callbacks[k]:
                    cls._callbacks[k].append(v)

    def __init__(self, _callbacks=None, logger=None, logging_config=None):
        if logger is not None:
            self.logger = logger
        else:
            try:
                self.set_logger(
                    log_dir=logging_config.logs_dir,
                    log_stream_level=logging_config.log_stream_level,
                    log_file_level=logging_config.log_file_level,
                )
            except Exception as e:
                print(f"There has been error when creating logger in callbacks: {e}")
                self.logger = None

        self._callbacks = get_default_callbacks()

        if _callbacks is not None:
            self.add_callbacks(_callbacks)

    def add_callback(self, event: str, callback):

        assert (
            event in self._callbacks
        ), f"event '{event}' not found in callbacks {self._callbacks}"
        
        if isinstance(callback, list):
            for _callback in callback:
                assert callable(_callback), RuntimeError(f"{event} must be callable")
                self._callbacks[event].append(_callback)
        else:
            assert callable(callback), RuntimeError(f"{event} must be callable")
            self._callbacks[event].append(callback)

    def add_callbacks(self, callbacks):
        for event, callback in callbacks.items():
            self.add_callback(event, callback)

    def set_callback(self, event: str, callback):

        assert (
            event in self._callbacks
        ), f"event '{event}' not found in callbacks {self._callbacks}"
        assert callable(callback), RuntimeError(f"{event} must be callable")

        self._callbacks[event] = callback

    def run_callbacks(self, event: str, *args, **kwargs):
        for callback in self._callbacks.get(event, []):
            callback(*args, **kwargs)