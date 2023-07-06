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
        action (int): index of chosen action
    '''
    epsilon = 0 if is_trained else 1
    action, _, _ = q_choose_number(
        numbers,
        oppo_numbers,
        is_first,
        odd_flag,
        epsilon=epsilon,
        soft_bound=CARD_NUM / 2
    )
    return action

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
        score, r_card_num = 0, CARD_NUM
        numbers = [list(range(1, CARD_NUM + 1)), list(range(1, CARD_NUM + 1))]
        round_winner = int(np.random.randint(PLAYER_NUM))
        game_over = False

        while not game_over:
            first, second = round_winner, 1 - round_winner
            r_card_num -= 1
            first_action = choose_number(
                numbers[first],
                numbers[second],
                first == TRAINED
            )
            second_action = choose_number(
                numbers[second],
                numbers[first],
                second == TRAINED,
                is_first=False,
                odd_flag=numbers[first][first_action] % 2 != 0
            )
            first_number = numbers[first].pop(first_action)
            second_number = numbers[second].pop(second_action)

            game_over = True
            if r_card_num == 1 and numbers[first] == numbers[second]:
                score += 0
            elif numbers[first][0] > numbers[second][-1]:
                score += (2 * first - 1) * (r_card_num)
            elif numbers[second][0] > numbers[first][-1]:
                score += (2 * second - 1) * (r_card_num)
            else:
                game_over = False

            if first_number > second_number:
                score -= 1
                round_winner = first
            elif first_number < second_number:
                score += 1
                round_winner = second

            if game_over:
                break

        if score != 0:
            wins[int(score > 0)] += 1

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
