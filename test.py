import os
import sys
import re
import subprocess
import logging as log

from tqdm import tqdm

log.basicConfig(level=log.INFO, filename='./logs/statistics.log', format='%(message)s')


def compute_stats(first, opponent, iterations = 5):
    if first:
        command_string = f'java -jar ManKalah.jar "python3 main_mcts.py" "java -jar {opponent}.jar"'
        heading = f'BOT vs {opponent}'
    else:
        command_string = f'java -jar ManKalah.jar "java -jar {opponent}.jar" "python3 main_mcts.py"'
        heading = f'{opponent} vs BOT'

    wins, time = 0, 0
    idx_win = 0 if first else 2
    idx_time = 1 if first else 3

    for i in tqdm(range(iterations), desc = heading):
        s = subprocess.run([command_string], stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell = True)
        numbers = re.findall("\d+", str(s.stdout))

        # wins for the bot
        wins += int(numbers[idx_win])
        # time per move for the bot
        time += int(numbers[idx_time])

    avg_time = time / iterations

    log.info(f'******** {heading} ********')
    log.info(f'WINS (for the bot): {wins} / {iterations} TIME: {avg_time} milliseconds\n')
    print(f'******** {heading} ********')
    print(f'WINS (for the bot): {wins} / {iterations} TIME: {avg_time} milliseconds\n')


if __name__ == '__main__':
    num_iters = 50

    compute_stats(True, 'MKRefAgent', num_iters)
    compute_stats(False, 'MKRefAgent', num_iters)

    compute_stats(True, 'Group2Agent', num_iters)
    compute_stats(False, 'Group2Agent', num_iters)

    compute_stats(True, 'JimmyPlayer', num_iters)
    compute_stats(False, 'JimmyPlayer', num_iters)