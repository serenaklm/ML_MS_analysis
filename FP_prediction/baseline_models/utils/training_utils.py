import copy
import torch
import torch.nn as nn 

def freeze_model(model):
    # Freeze all the layers
    for param in model.parameters():
        param.requires_grad = False

def set_submodule_by_path(root_module: nn.Module, path: str, new_module: nn.Module):

    parts = path.split(".")
    submodule = root_module
    
    for name in parts[:-1]:
        if name.isdigit():
            index = int(name)
            submodule = submodule[index]
        else:
            submodule = getattr(submodule, name)
    
    final_name = parts[-1]
    if final_name.isdigit():
        index = int(final_name)
        submodule[index] = new_module
    else:
        # Final part is a normal attribute
        setattr(submodule, final_name, new_module)

def replace_non_dynamic_linear(model: nn.Module) -> nn.Module:

    all_names, all_modules = [], []

    for name, module in model.named_modules():

        if isinstance(module, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
            
            # 1. Get old layer parameters
            in_features = module.in_features
            out_features = module.out_features
            bias = (module.bias is not None)
            module_copy = copy.deepcopy(module)

            old_weight = module_copy.weight.detach()
            old_bias = module_copy.bias.detach() if bias else None
            
            # old_weight = module.weight.detach()
            # old_bias = module.bias.detach() if bias else None
            
            # 2. Create new layer
            new_module = nn.modules.linear.Linear(in_features, out_features, bias = bias)
            
            # 3. Copy parameters
            with torch.no_grad():
                new_module.weight.copy_(old_weight)
                if bias:
                    new_module.bias.copy_(old_bias)

            # 4. Add it to list 
            all_names.append(name)
            all_modules.append(new_module)

    # Iterate through 
    for idx, name in enumerate(all_names):
        print(name)
        set_submodule_by_path(model, name, all_modules[idx])

    return model.train()

def replace_MS_encoder_layers(model: nn.Module):

    n_layers = len(model.MS_encoder.layers)

    for i in range(n_layers):
            
        module = model.MS_encoder.layers[i].self_attn.out_proj

        bias = (module.bias is not None)
        in_features = module.in_features
        out_features = module.out_features
        
        old_weight = module.weight.detach()
        old_bias = module.bias.detach() if bias else None
        
        # 2. Create new layer
        new_module = nn.Linear(in_features, out_features, bias = bias)
        
        # 3. Copy parameters
        with torch.no_grad():
            new_module.weight.copy_(old_weight)
            if bias:
                new_module.bias.copy_(old_bias)

        model.MS_encoder.layers[i].self_attn.out_proj = new_module