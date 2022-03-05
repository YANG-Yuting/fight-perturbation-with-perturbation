import torch
import torch.nn as nn


def hook(module, grad_input, grad_output):
    print('grad_input: ', grad_input)
    print('grad_output: ', grad_output)
    return grad_input[0] * 0, grad_input[1] * 0, grad_input[2] * 0,


x = torch.tensor([[1., 2., 10.]], requires_grad=True)
module = nn.Linear(3, 1)
handle = module.register_backward_hook(hook)
y = module(x)
y.backward()
print('module_bias: ', module.bias.grad)
print('x: ', x.grad)
print('module_weight: ', module.weight.grad)

handle.remove()