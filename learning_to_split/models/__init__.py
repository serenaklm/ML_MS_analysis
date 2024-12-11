from .MLP import MLP

# Define the buildin_models list
from .build import ModelFactory
builtin_models = list(ModelFactory.registry.keys())