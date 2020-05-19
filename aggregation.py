
import copy
import torch


def get_aggregation_func(alg):
    """FedAvg만 구현되어 있음"""

    if 'fedavg' == alg:
        def aggregate_models(w):
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[k] += w[i][k]
                w_avg[k] = torch.div(w_avg[k], len(w))
            return copy.deepcopy(w_avg)
        return aggregate_models
    else:
        raise NotImplementedError
