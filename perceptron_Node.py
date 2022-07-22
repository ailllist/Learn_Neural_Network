import numpy

class Node:
    
    def __init__(self, value, stage):
        self.value = value  # input value
        self.stage = stage  # order of layer
        self.next_node = []
        
    def add_next(self, node, weights):
        self.next_node.append([node, weights])
        
    def __call__(self):
        print("value: ", self.value)
        print("stage: ", self.stage)
        print("next_node: ", self.next_node)
        
if __name__ == "__main__":
    
    input_data = [1, 2]
    stage = 1
    for i in range(stage):
        for j in input_data:
            tmp_node = Node(j, i)
            