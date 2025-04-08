from .MS_binned_model import MSBinnedModel

# Define the buildin_models list
from .build import ModelFactory
builtin_models = list(ModelFactory.registry.keys())