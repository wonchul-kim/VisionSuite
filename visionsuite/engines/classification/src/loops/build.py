from visionsuite.engines.classification.utils.registry import LOOPS

def build_loop(**loop_config):
    loop = LOOPS.get(loop_config['type'], case_sensitive=True)()

    return loop