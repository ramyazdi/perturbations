from parsing import *
from neuro_nlp.weighted_graph import *

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

class PerturbatedPosTagger(nn.Module):
    def __init__(self,biaffine_net,alpha=0.1):
        super(PerturbatedPosTagger, self).__init__()
        self.non_perturbated_biaffine = biaffine_net
        self.alpha = nn.Parameter(torch.Tensor([alpha]))
        # freezing the non-perturbated network's parameters
        for name, param in self.non_perturbated_biaffine.named_parameters():
            #print(name," : ",param.size())
            if param.requires_grad and param.is_leaf  and 'dense' in name: #and 'word_embedd' not in name:
                param.requires_grad = False
                param.is_perturbated_ = True

            else:
                print ("Layer "+name+" won't get perturbated!")
                param.is_perturbated_ = False


    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0):

        #perturbating all parametrs of trained model which have is_perturbated_ flag as True (think of an option to pass not_prtur_net by deep_copy)
        with torch.no_grad():
            for name, param in self.non_perturbated_biaffine.named_parameters():
                if param.is_perturbated_:
                    epsilon = torch.randn(param.size(),device=device) * self.alpha + 1
                    param.mul_(epsilon)

            # [batch, length, tag_space]
            loss, corr, preds = self.non_perturbated_biaffine.loss(input_word, input_char,target, mask=mask, length=length, hx=hx,leading_symbolic=leading_symbolic)

        return loss, corr, preds