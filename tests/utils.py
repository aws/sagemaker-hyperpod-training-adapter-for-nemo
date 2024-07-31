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
