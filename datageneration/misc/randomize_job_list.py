import numpy as np
import sys

lines = []
with open("job_list.txt") as f:
    for line in f:
        lines.append(line)

lines = np.array(lines)
randomized_list = np.random.permutation(lines)
randomized_list = randomized_list

for i in range(len(randomized_list)):
    sys.stdout.write(randomized_list[i])
