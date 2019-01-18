import numpy as np
envs = ['beamrider', 'breakout', 'enduro', 'hero', 'mspacman', 'pong', 'qbert', 'seaquest', 'spaceinvaders', 'videopinball']

for e in envs:
    reader = open('../checkpoints/' + e + "_bc_results.txt")
    sum_returns = 0.0
    count = 0.0
    for line in reader:
        sum_returns += float(line)
        count += 1.0
    print(e,sum_returns / count)
