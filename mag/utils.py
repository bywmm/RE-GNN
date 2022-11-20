import torch
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter, segment_csr, gather_csr
from texttable import Texttable

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))

def weighted_degree(index, edge_weight=None, num_nodes=None, dtype=None):

    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    if edge_weight is None:
        edge_weight = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, edge_weight)

def get_self_loop_index(num_node):
    self_loop_index = torch.arange(0, num_node, dtype=torch.long)
    self_loop_index = self_loop_index.unsqueeze(0).repeat(2, 1)
    return self_loop_index

def softmax(src: Tensor, index, ptr=None,
            num_nodes=None, temperature=1.0):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    src /= temperature
    out = src - src.max()
    out = out.exp()

    if ptr is not None:
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)

class MsgNorm(torch.nn.Module):
    def __init__(self, learn_msg_scale=False):
        super(MsgNorm, self).__init__()

        self.msg_scale = torch.nn.Parameter(torch.Tensor([1.0]),
                                            requires_grad=learn_msg_scale)
        self.reset_parameters()

    def forward(self, x, msg, p=2):
        msg = F.normalize(msg, p=p, dim=1)
        x_norm = x.norm(p=p, dim=1, keepdim=True)
        msg = msg * x_norm * self.msg_scale
        return msg

    def reset_parameters(self):
        torch.nn.init.ones_(self.msg_scale)

def args_print(args):
    _dict = vars(args)
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())