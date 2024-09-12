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
    def get_model(cls, config_dict = None, splitter = False, predictor = False) -> nn.Module:
        
        """
            Return a model (nn.Module) based on the input:
                args: argparse containing the configuration
                splitter: whether this is the splitter or not
                predictor: whether this is the predictor or not        
        """

        if config_dict["model_name"] not in cls.registry:
            model_name = config_dict["model_name"]
            raise ValueError(f"Model {model_name} does not exist in the registry")
        
        assert (int(splitter) + int(predictor)) == 1, "Either splitter or predictor must be true."

        exec_class = cls.registry[config_dict["model_name"]]

        if splitter:
            model = exec_class(config_dict, is_splitter = True, n_classes = 2)
        else:
            if not predictor: raise Exception("Model needs to be either a predictor or a splitter")
            model = exec_class(config_dict, is_splitter = False, n_classes = config_dict["FP_dim_mapping"][config_dict["FP_type"]]) # The predictor

        return model.to(config_dict["device"])