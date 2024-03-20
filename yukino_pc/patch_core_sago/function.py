import torch 
import torchvision
##################################################################

##################################################################
def print_model(module, name="model", depth=0):
    if len(list(module.named_children())) == 0:
        print(f"{' ' * depth} {name}: {module}")
    else:
        print(f"{' ' * depth} {name}: {type(module)}")

    for child_name, child_module in module.named_children():
        if isinstance(module, torch.nn.Sequential):
            child_name = f"{name}[{child_name}]"
        else:
            child_name = f"{name}.{child_name}"
        print_model(child_module, child_name, depth + 1)




def timm_extract(model, targets, inputs):
    feature = None

    def forward_hook(model, inputs, outputs):
        global features
        features = outputs.detach().clone()

    handle = targets.register_forward_hook(forward_hook)

    model.eval()
    model(inputs)

    handle.remove()

    return features
