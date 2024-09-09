import sys


def _strip_mp_params_helper(args):
    if "--mp_parameters" not in args:
        return args

    mp_params_idx = args.index("--mp_parameters")

    for i in range(mp_params_idx + 1, len(args)):
        if args[i].startswith("-"):
            return args[0:mp_params_idx] + args[i:]
    return args[0:mp_params_idx]


def setup_args_for_sm():
    """
    Set up command line args as expected by training adaptor
    when running using sagemaker jobs.
    """
    sys.argv = _strip_mp_params_helper(sys.argv)
