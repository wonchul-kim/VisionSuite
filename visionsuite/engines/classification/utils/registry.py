from visionsuite.engines.utils.registry import RUNNERS as ROOT_RUNNERS
from visionsuite.engines.utils.registry import Registry


RUNNERS = Registry('runners', parent=ROOT_RUNNERS, 
                   scope='visionsuite.engines.classification.src',
                   locations=['visionsuite.engines.classification.src.runners']
        )
