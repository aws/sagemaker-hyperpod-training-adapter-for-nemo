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

from types import SimpleNamespace


class NestedDotMap(SimpleNamespace):
    """
    Dictionary-like class for creating nested attributes that can be accesssed using dot notation. For example:

    dot_map = NestedDotMap({
        'country': {
            'city': 'London'
        }
    })

    dot_map.country.city == "London"
    """

    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)

        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedDotMap(value))
            else:
                self.__setattr__(key, value)
