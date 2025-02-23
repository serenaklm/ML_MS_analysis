
def freeze_model(model):
    # Freeze all the layers
    for param in model.parameters():
        param.requires_grad = False

