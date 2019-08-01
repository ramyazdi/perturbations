import torch

class weighted_graph:
    @staticmethod
    def create_weighted_garph(predicted_heads_tensor_list):
        weighted_graph = torch.zeros(predicted_heads_tensor_list[0].size()+(predicted_heads_tensor_list[0].size()[1],))
        for predicted_heads_tensor in predicted_heads_tensor_list:
            for i in range(predicted_heads_tensor.size()[0]):
                weighted_graph[i, predicted_heads_tensor[i].to(dtype=torch.long), list(range(predicted_heads_tensor.size()[1]))] += 1

        return weighted_graph

