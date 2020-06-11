
import copy
import torch


def get_aggregation_func(alg):

    if 'fedavg' == alg:
        def aggregate_models(w):
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[k] += w[i][k]
                w_avg[k] = torch.div(w_avg[k], len(w))
            return copy.deepcopy(w_avg)
        return aggregate_models
  
    elif 'fedpr' == alg:
        def aggregate_models(w):
            w_avg = copy.deepcopy(w[0])
            w_exist = dict()
            
            for k in w_avg.keys():
                w_exist[k] = torch.zeros_like(w_avg[k]).to(w_avg[k].device)
                for i in range(1, len(w)):
                    w_exist[k] += (w[i][k] != 0).float()
                    w_avg[k] += w[i][k]
                    w_exist[k][w_exist[k] == 0] = 1
                w_avg[k] = torch.div(w_avg[k], w_exist[k])
            return copy.deepcopy(w_avg)
        return aggregate_models
    
    else:
        raise NotImplementedError