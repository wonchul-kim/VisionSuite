from visionsuite.engines.utils.torch_utils.dist import init_distributed_mode
from visionsuite.engines.utils.torch_utils.utils import set_torch_deterministic, get_device, parse_device_ids

def set_variables(args):
    args.device_ids = parse_device_ids(args.device_ids)

    init_distributed_mode(args)

    set_torch_deterministic(args.use_deterministic_algorithms)
    args.device = get_device(args.device)
