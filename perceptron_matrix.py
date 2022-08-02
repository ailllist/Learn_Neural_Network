import numpy as np

def ReLU(inp):
    return max(0, inp)

def stage(input_data, weights, bias):
    input_data = np.array(input_data).T
    weights = np.array(weights).T
    signal_total = input_data@weights + bias
    try:
        signal_f = map(lambda x: ReLU(x), signal_total)
        return list(signal_f)
    except:
        signal_f = ReLU(signal_total)
        return signal_f

def back_propagation():
    pass

stage_res = [1, 1]
stage_info = [[[[0.5, 0.5], [0.1, 0.3]], 0.6], [[0.1, 0.5], 1]]
for i in stage_info:
    stage_res = stage(stage_res, i[0], bias=i[1])
