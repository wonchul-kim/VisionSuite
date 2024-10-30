from visionsuite.engines.utils.registry import RUNNERS as ROOT_RUNNERS
from visionsuite.engines.utils.registry import MODELS as ROOT_MODELS
from visionsuite.engines.utils.registry import OPTIMIZERS as ROOT_OPTIMIZERS
from visionsuite.engines.utils.registry import DATASETS as ROOT_DATASETS
from visionsuite.engines.utils.registry import DATALOADERS as ROOT_DATALOADERS
from visionsuite.engines.utils.registry import LOSSES as ROOT_LOSSES
from visionsuite.engines.utils.registry import SCHEDULERS as ROOT_SCHEDULERS
from visionsuite.engines.utils.registry import LOOPS as ROOT_LOOPS
from visionsuite.engines.utils.registry import PIPELINES as ROOT_PIPELINES
from visionsuite.engines.utils.registry import Registry


scope = 'visionsuite.engines.classification.src'
RUNNERS = Registry('runners', parent=ROOT_RUNNERS, 
                   scope=scope,
                   locations=['visionsuite.engines.classification.src.runners']
        )

MODELS = Registry('models', parent=ROOT_MODELS, 
                   scope=scope,
                   locations=['visionsuite.engines.classification.src.models']
        )

OPTIMIZERS = Registry('optimizers', parent=ROOT_OPTIMIZERS, 
                   scope=scope,
                   locations=['visionsuite.engines.classification.src.optimizers']
        )

DATASETS = Registry('datasets', parent=ROOT_DATASETS, 
                   scope=scope,
                   locations=['visionsuite.engines.classification.src.datasets']
        )

DATALOADERS = Registry('dataloaders', parent=ROOT_DATALOADERS, 
                   scope=scope,
                   locations=['visionsuite.engines.classification.src.dataloaders']
        )

LOSSES = Registry('losses', parent=ROOT_LOSSES, 
                   scope=scope,
                   locations=['visionsuite.engines.classification.src.losses']
        )

SCHEDULERS = Registry('schedulers', parent=ROOT_SCHEDULERS, 
                   scope=scope,
                   locations=['visionsuite.engines.classification.src.schedulers']
        )

LOOPS = Registry('loops', parent=ROOT_LOOPS, 
                   scope=scope,
                   locations=['visionsuite.engines.classification.src.loops']
        )

PIPELINES = Registry('pipelines', parent=ROOT_PIPELINES, 
                   scope=scope,
                   locations=['visionsuite.engines.classification.src.pipelines']
        )
