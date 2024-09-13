import random

# Transition structure copied from Durrant et al. (2011):
transition_structure = {
    (1,1): 4,
    (1,2): 3,
    (1,3): 2,
    (1,4): 1,
    (1,5): 5,
    (2,1): 5,
    (2,2): 4,
    (2,3): 3,
    (2,4): 2,
    (2,5): 1,
    (3,1): 3,
    (3,2): 2,
    (3,3): 1,
    (3,4): 5,
    (3,5): 4,
    (4,1): 1,
    (4,2): 5,
    (4,3): 4,
    (4,4): 3,
    (4,5): 2,
    (5,1): 2,
    (5,2): 1,
    (5,3): 5,
    (5,4): 4,
    (5,5): 3
}

def get_sequence():
    start = [random.randint(1,5),random.randint(1,5)]
    for i in range(50):
        num = random.uniform(0, 1)
        if num > 0.1:
            next_val = transition_structure[tuple(start[-2:])]
        else:
            if 0 < num < 0.02:
                next_val = 1
            if 0.02 < num < 0.04:
                next_val = 2
            if 0.04 < num < 0.06:
                next_val = 3
            if 0.06 < num < 0.08:
                next_val = 4
            if 0.08 < num < 0.1:
                next_val = 5
        start.append(next_val)
    return ','.join([str(i) for i in start])
