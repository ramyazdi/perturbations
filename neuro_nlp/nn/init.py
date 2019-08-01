__author__ = 'max'
#from torch.autograd import Variable

def assign_tensor(tensor, val):
    """
    copy val to tensor
    Args:
        tensor: an n-dimensional torch.Tensor
        val: an n-dimensional torch.Tensor to fill the tensor with

    Returns:

    """
    #if isinstance(tensor, Variable):
    #    assign_tensor(tensor.data, val)
    #    return tensor
    #return tensor[1:val.size()[0],].data.copy_(val[1:])
    return tensor.data.copy_(val)
