'''Play game with trained computer'''
import argparse
import numpy as np

from train.train import CARD_NUM, PLAYER_NUM, q_choose_number, read_q_table
from simul.simul import simulate

PLAYER, COMPUTER = 0, 1
NAMES = ['Player', 'Computer']

def choose_number(numbers, oppo_numbers, is_player=False, is_first=True, odd_flag=False):
    '''
    Choose a number from number list or Q-table.
    Args:
        numbers (list): Card number list
        oppo_numbers (list): Card number list of opponent
        is_player (bool): Indicates whether caller is the player or computer
        is_first (bool): Indicates whether player is the first or not
        odd_flag (bool): Indicates whether number of opponent is odd or not
    Returns:
        Tuple containing the following elements:
        action (int): index of chosen action
    '''
    if is_player:
        while True:
            try:
                chosen_number = int(input("Choose a number from the available numbers: "))
                if chosen_number not in numbers:
                    raise ValueError
                break
            except ValueError:
                print("Invalid choice. Please choose a number from the available numbers.")
        action = numbers.index(chosen_number)
    else:
        action, _, _ = q_choose_number(
            numbers,
            oppo_numbers,
            is_first,
            odd_flag,
            soft_bound=CARD_NUM / 2
        )
    return action

def play_game():
    '''
    Play game with computer
    Args:
        None
    Returns:
        None
    '''
    scores = [0] * PLAYER_NUM
    numbers = [list(range(1, CARD_NUM + 1)), list(range(1, CARD_NUM + 1))]
    round_winner = int(np.random.randint(PLAYER_NUM))
    odd_msg = ['even', 'odd']
    r_card_num = CARD_NUM
    game_over = False

    while not game_over:
        first, second = round_winner, 1 - round_winner
        r_card_num -= 1
        print(f'[Round {CARD_NUM - r_card_num}]')
        print("Computer's available numbers:\t", numbers[COMPUTER])
        print("Player's available numbers:\t", numbers[PLAYER])
        first_action = choose_number(
            numbers[first],
            numbers[second],
            first == PLAYER
        )
        odd_flag = int(numbers[first][first_action] % 2 != 0)
        print(f'{NAMES[first]} has chosen an {odd_msg[odd_flag]} number.')
        second_action = choose_number(
            numbers[second],
            numbers[first],
            second == PLAYER,
            is_first=False,
            odd_flag=bool(odd_flag)
        )
        first_number = numbers[first].pop(first_action)
        second_number = numbers[second].pop(second_action)
        print(f'Computer has chosen the number {first_number if first == COMPUTER else second_number}.')

        game_over = True
        if r_card_num == 1 and numbers[first] == numbers[second]:
            scores[first] += 0
        elif numbers[first][0] > numbers[second][-1]:
            scores[first] += r_card_num
        elif numbers[second][0] > numbers[first][-1]:
            scores[second] += r_card_num
        else:
            game_over = False

        if first_number > second_number:
            print(f'{NAMES[first]} won!')
            scores[first] += 1
            round_winner = first
        elif first_number < second_number:
            print(f'{NAMES[second]} won!')
            scores[second] += 1
            round_winner = second
        else:
            print("It's a tie!")
        print()

    print("Game over!")
    print("Player's score:", scores[PLAYER])
    print("Computer's score:", scores[COMPUTER])

    if scores[PLAYER] > scores[COMPUTER]:
        print("Congratulations! You are the ultimate winner!")
    elif scores[PLAYER] < scores[COMPUTER]:
        print("Computer is the ultimate winner!")
    else:
        print("The final result is a tie!")


parser = argparse.ArgumentParser(description='Play game with trained computer')
parser.add_argument('--input', type=str, default='q_table.txt', help='Q-table input file path')
parser.add_argument('-n', '--number', type=int, default=0, help='Number of simulation')
parser.add_argument('-s', '--sim', action='store_true', help='Run a simulation')
args = parser.parse_args()

read_q_table(args.input)
if args.sim:
    simulate(args.number)
else:
    play_game()
