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

input_data = [1, 1]

stage1 = stage(input_data, [[0.5, 0.5], [0.1, 0.3]], bias=0.6)
print(stage1)
stage2 = stage(stage1, [0.1, 0.5], bias=1)
print(stage2)