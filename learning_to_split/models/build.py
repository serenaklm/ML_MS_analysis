import argparse
from typing import Callable

import torch.nn as nn

class ModelFactory:
    
    """ The factory class for creating models """
    
    registry = {}
    """ Internal registry for available models """

    @classmethod
    def register(cls, name: str) -> Callable:
        """ Class method to register Executor class to the internal registry.

        Args:
            name (str): The name of the executor.

        Returns:
            The model class itself.
        """

        def inner_wrapper(wrapped_class: nn.Module) -> Callable:
            if name in cls.registry:
                print(f'Model {name} already exists. Will replace it')

            cls.registry[name] = wrapped_class

            return wrapped_class

        return inner_wrapper
    
    @classmethod
    def get_model(cls, config: dict, splitter = False, predictor = False) -> nn.Module:
        
        model_name = config["model"]["name"]
        if model_name not in cls.registry:
            raise ValueError(f"Model {model_name} does not exist in the registry")
        
        assert (int(splitter) + int(predictor)) == 1, "Either splitter or predictor must be true."

        exec_class = cls.registry[model_name]

        if splitter:
            model = exec_class(is_splitter = True, output_dim = 2, config = config)
        else: 
            model = exec_class(is_splitter = False, config = config) # The predictor

        return model