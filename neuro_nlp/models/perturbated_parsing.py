from parsing import *
from neuro_nlp.weighted_graph import *

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

class PerturbatedParser(nn.Module):
    def __init__(self,biaffine_net,alpha=0.1):
        super(PerturbatedParser, self).__init__()
        self.non_perturbated_biaffine = biaffine_net
        self.alpha = nn.Parameter(torch.Tensor([alpha]))  #torch.randn(1))
        # freezing the non-perturbated network's parameters
        for i, param in enumerate(self.non_perturbated_biaffine.parameters()):
            if param.requires_grad and param.is_leaf:
                param.requires_grad = False


    def forward(self,input_word, input_char, input_pos, heads, arc_tags ,mask=None, length=None, hx=None,return_loss = True,non_perturbated_biaffine=None):

        #perturbating all parametrs of trained model
        if return_loss:

            for i, param in enumerate(non_perturbated_biaffine.parameters()):
                epsilon = torch.randn(param.size(),device=device) * self.alpha + 1
                param.mul_(epsilon)

            return non_perturbated_biaffine.loss(input_word, input_char, input_pos, heads, arc_tags, mask, length, hx)

        else:
            for i, param in enumerate(self.non_perturbated_biaffine.parameters()):
                epsilon = torch.randn(param.size()).to(device) * self.alpha + 1
                param.mul_(epsilon)

            return self.non_perturbated_biaffine.forward(input_word, input_char, input_pos, mask, length, hx)

    def decode_mst_perturabted(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0,labeled = False,perturbated_K=10,not_perturbated_network=None):
        '''
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in arc_tag alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and arc_tags.

        '''
        # out_arc shape [batch, length, length]
        predicted_heads_list = []
        with torch.no_grad():
            for index in range(perturbated_K):
                loss_arc_sum_t, loss_arc_t_tensor, gold_indicies,loss_arc_tag = self.forward(input_word, input_char, input_pos, heads=torch.ones(input_pos.size()).long().to(device),#fake for test
                                                                                        arc_tags=None,
                                                                                       mask=mask, length=length,
                                                                                       return_loss=True,
                                                                                       non_perturbated_biaffine=not_perturbated_network)  # loss arc is the same tensor before summing

                out_arc = F.log_softmax(loss_arc_t_tensor, dim=1)
                # mask invalid position to -inf for log_softmax
                if mask is not None:
                    minus_inf = -1e8
                    minus_mask = (1 - mask) * minus_inf
                    out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

                energy = torch.exp(out_arc)
                predicted_heads, _ = parse.decode_MST_tensor(energy, length, leading_symbolic=0, labeled=False)
                predicted_heads = torch.from_numpy(predicted_heads)
                predicted_heads_list.append((predicted_heads))

                # arc_tag_h = arc_tag_h.unsqueeze(2).expand(batch, max_len, max_len, arc_tag_space).contiguous()
                # arc_tag_c = arc_tag_c.unsqueeze(1).expand(batch, max_len, max_len, arc_tag_space).contiguous()
                # compute output for arc_tag [batch, length, length, num_arcs]
                # out_arc_tag = self.bilinear(arc_tag_h, arc_tag_c)





        weighted_graph_tensor = weighted_graph.create_weighted_garph(predicted_heads_list)
        final_predicted_heads, _ = parse.decode_MST_tensor(weighted_graph_tensor, length, leading_symbolic=0,labeled=False)

        return from_numpy(final_predicted_heads)



def decode_mst_perturabted_backup(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0,
                           labeled=False, perturbated_K=10):
    '''
    Args:
        input_word: Tensor
            the word input tensor with shape = [batch, length]
        input_char: Tensor
            the character input tensor with shape = [batch, length, char_length]
        input_pos: Tensor
            the pos input tensor with shape = [batch, length]
        mask: Tensor or None
            the mask tensor with shape = [batch, length]
        length: Tensor or None
            the length tensor with shape = [batch]
        hx: Tensor or None
            the initial states of RNN
        leading_symbolic: int
            number of symbolic labels leading in arc_tag alphabets (set it to 0 if you are not sure)

    Returns: (Tensor, Tensor)
            predicted heads and arc_tags.

    '''
    # out_arc shape [batch, length, length]
    out_arc_sum = torch.Tensor(input_pos.size(0), input_pos.size(1), input_pos.size(1)).to(device)
    with torch.no_grad():
        for index in range(perturbated_K):
            out_arc, out_arc_tag, mask, length = self.forward(input_word, input_char, input_pos, heads=None, arc_tags=None,
                                                 mask=mask, length=length, hx=hx, return_loss=False)
            out_arc_sum += out_arc

    # out_arc_tag shape [batch, length, arc_tag_space]
    # arc_tag_h, arc_tag_c = out_arc_tag
    # batch, max_len, arc_tag_space = arc_tag_h.size()


    batch, max_len, _ = out_arc_sum.size()

    # compute lengths
    if length is None:
        if mask is None:
            length = [max_len for _ in range(batch)]
        else:
            length = mask.data.sum(dim=1).long().cpu().numpy()

    # arc_tag_h = arc_tag_h.unsqueeze(2).expand(batch, max_len, max_len, arc_tag_space).contiguous()
    # arc_tag_c = arc_tag_c.unsqueeze(1).expand(batch, max_len, max_len, arc_tag_space).contiguous()
    # compute output for arc_tag [batch, length, length, num_arcs]
    # out_arc_tag = self.bilinear(arc_tag_h, arc_tag_c)

    # mask invalid position to -inf for log_softmax
    if mask is not None:
        minus_inf = -1e8
        minus_mask = (1 - mask) * minus_inf
        out_arc_sum = out_arc_sum + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

    # loss_arc shape [batch, length, length]
    loss_arc = F.log_softmax(out_arc_sum, dim=1)
    # loss_arc_tag shape [batch, length, length, num_arcs]
    # loss_arc_tag = F.log_softmax(out_arc_tag, dim=3).permute(0, 3, 1, 2)
    # [batch, num_arcs, length, length]
    energy = torch.exp(loss_arc)  # .unsqueeze(1)) #+ loss_arc_tag)
    heads, arc_tags = parse.decode_MST_tensor(energy, length, leading_symbolic=leading_symbolic,
                                              labeled=labeled)  # .data.cpu().numpy()
    heads = from_numpy(heads)
    # arc_tags = from_numpy(arc_tags)

    return heads  # ,arc_tags