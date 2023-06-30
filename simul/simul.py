'''Simulate trained computer'''
import argparse
import numpy as np
from tqdm import tqdm

from train.train import CARD_NUM, PLAYER_NUM, q_choose_number, read_q_table

DEFAULT, TRAINED = 0, 1

def choose_number(numbers, oppo_numbers, is_trained, is_first=True, odd_flag=False):
    '''
    Choose a number randomly, or from the Q-table.
    Args:
        numbers (list): Card number list
        oppo_numbers (list): Card number list of opponent
        is_trained (bool): Indicates whether caller is the trained or not
        is_first (bool): Indicates whether player is the first or not
        odd_flag (bool): Indicates whether number of opponent is odd or not
    Returns:
        Tuple containing the following elements:
        chosen_number (int): chosen number
        numbers (list): number list after choosing
    '''
    epsilon = 0 if is_trained else 1
    _, _, chosen_number, numbers, _ = q_choose_number(
        numbers,
        oppo_numbers,
        is_first,
        odd_flag,
        epsilon=epsilon,
        soft_bound=CARD_NUM / 2
    )
    return chosen_number, numbers

def simulate(simul_num):
    '''
    Simulate game and print result
    Args:
        simul_num (int): Number of simulation
    Returns:
        None
    '''
    wins = [0] * PLAYER_NUM
    for _ in tqdm(range(simul_num), mininterval=1):
        scores = [0] * PLAYER_NUM
        numbers = [list(range(1, CARD_NUM + 1)), list(range(1, CARD_NUM + 1))]
        first = int(np.random.randint(PLAYER_NUM))
        second = 1 - first
        game_over = False

        for round_num in range(1, CARD_NUM + 1):
            prev_numbers = numbers[first][:]
            first_number, numbers[first] = choose_number(
                numbers[first],
                numbers[second],
                first == TRAINED
            )
            odd_flag = first_number % 2 != 0
            second_number, numbers[second] = choose_number(
                numbers[second],
                prev_numbers,
                second == TRAINED,
                is_first=False,
                odd_flag=odd_flag
            )

            if first_number > second_number:
                scores[first] += 1
            elif first_number < second_number:
                scores[second] += 1

            for i in range(PLAYER_NUM):
                if min(numbers[i]) >= max(numbers[1 - i]):
                    scores[i] += CARD_NUM - round_num - int(min(numbers[i]) == max(numbers[1 - i]))
                    game_over = True
                    break
            if game_over:
                break
            first, second = second, first

        if scores[DEFAULT] > scores[TRAINED]:
            wins[DEFAULT] += 1
        elif scores[DEFAULT] < scores[TRAINED]:
            wins[TRAINED] += 1

    if simul_num > 0:
        print('[Simulation result]')
        print(f'Computer (trained):\t{100 * wins[TRAINED] / simul_num:.2f}%')
        print(f'Computer (not trained):\t{100 * wins[DEFAULT] / simul_num:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate trained computer')
    parser.add_argument('-n', '--number', type=int, help='Number of simulation')
    parser.add_argument('--input', type=str, default='q_table.txt', help='Q-table input file path')
    args = parser.parse_args()
    read_q_table(args.input)
    simulate(simul_num=args.number)
