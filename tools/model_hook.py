from yolox.models import YOLOX
from torch import nn
import torch

class YOLOXHook((nn.Module)):

    def __init__(self, model, fp16=False, img_shape=(640, 640)):
        super().__init__()
        self.fp16 = fp16
        self.model = model.half() if fp16 else model.float()
        self.hooks = {}
        self.img_shape = img_shape
        self.p6 = False

    def forward(self, x):
        return self.model(x)

    def register_preprocessing_hook(self):
        """Register hooks to convert uint8 to fp16/fp32"""
        if "preprocessing" in self.hooks:
            return
        self.hooks["preprocessing"] = self.register_forward_pre_hook(self._preprocessing_hook)

    def register_postprocessing_hook(self):
        if "postprocessing" in self.hooks:
            return
        self.hooks["postprocessing"] = self.register_forward_hook(self._postprocessing_hook)

    def register_io_hooks(self):
        """Register hooks for input and output processing."""
        self.register_preprocessing_hook()
        self.register_postprocessing_hook()

    def remove_hooks(self):
        """Remove hooks."""
        for _, hook in self.hooks.items():
            hook.remove()
        self.hooks.clear()

    @staticmethod
    def _preprocessing_hook(module, inputs):
        """Add preprocessing operations to be part of the model."""
        def _preprocess(x):
            x = x.half() if module.fp16 else x.float()
            return x
        # if not module.fp16:
        #     return inputs  # ls
        return tuple(_preprocess(inp) for inp in inputs)

    def _postprocessing_hook(self, module, inputs, outputs):
        """Add postprocessing operations to be part of the model."""
        # outputs = self._postprocess_strides(module, inputs, outputs)
        outputs = self._postprocessing_float(module, inputs, outputs)
        return

    @staticmethod
    def _postprocessing_float(module, inputs, outputs):
        """convert to fp32 if required."""
        if module.fp16:
            outputs = outputs.float()
        return outputs

    @staticmethod
    def _postprocess_strides(module, inputs, outputs):
        outputs = outputs[0]
        grids = torch.empty(1, 0, outputs.shape[1]-5, device=outputs.device)
        expanded_strides = []
        strides = torch.tensor([8, 16, 32], device=outputs.device) if not module.p6 else torch.tensor([8, 16, 32, 64], device=outputs.device)
        hsizes = module.img_shape[0] // strides 
        wsizes = module.img_shape[1] // strides

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            # Create meshgrid
            xv, yv = torch.meshgrid(torch.arange(wsize, device=outputs.device), torch.arange(hsize, device=outputs.device))
            grid = torch.stack((xv, yv), dim=2).reshape(1, -1, 2)
            grids = torch.cat((grids, grid), dim=1)
            # Create expanded strides
            shape = grid.shape[:2]
            expanded_strides.append(torch.full((*shape, 1), stride, dtype=torch.float16, device=outputs.device))
        
        expanded_strides = torch.cat(expanded_strides, dim=1)

        # Adjust outputs
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

def torch_postprocess_strides(outputs, img_size, p6=False):
    grids = torch.empty(1, 0, outputs.shape[1]-5, device=outputs.device)
    expanded_strides = []
    strides = torch.tensor([8, 16, 32], device=outputs.device) if not p6 else torch.tensor([8, 16, 32, 64], device=outputs.device)
    hsizes = img_size[0] // strides 
    wsizes = img_size[1] // strides

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        # Create meshgrid
        xv, yv = torch.meshgrid(torch.arange(wsize), torch.arange(hsize))
        grid = torch.stack((xv, yv), dim=2).reshape(1, -1, 2)
        grids = torch.cat((grids, grid), dim=1)
        # Create expanded strides
        shape = grid.shape[:2]
        expanded_strides.append(torch.full((*shape, 1), stride, dtype=torch.float16, device=outputs.device))
    
    expanded_strides = torch.cat(expanded_strides, dim=1)

    # Adjust outputs
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


    

def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))


    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


if __name__ == "__main__":
    import numpy as np
    
    #load tensor.npy
    np_input = np.load("tensor.npy")
    torch_input = torch.tensor(np_input)

    np_output = demo_postprocess(np_input, (640, 640))

    torch_output = torch_postprocess_strides(torch_input, (640, 640))
    
    print(np_output[0])
    print(torch_output[0].numpy())