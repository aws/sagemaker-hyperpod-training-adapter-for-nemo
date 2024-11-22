# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

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
    Set up command line args as expected by training adapter
    when running using sagemaker jobs.
    """
    sys.argv = _strip_mp_params_helper(sys.argv)
