"""Q-learning module"""
import argparse
import numpy as np
from tqdm import tqdm

CARD_NUM = 7
PLAYER_NUM = 2
NUM_STATES = 2 ** (2 + 2 * CARD_NUM)
FIRST, SECOND = 0, 1
TEMPERATURE = 0.5
ROUND_REWARD, FINAL_REWARD = 1, 5

def initialize_q_table():
    '''
    Initialize the Q-table.
    Args:
        None
    Returns:
        None
    '''
    for state in range(NUM_STATES):
        # is_first = True -> is_odd does not matter
        if state % 4 == 3:
            continue

        # allocate list with number of remaining card slots
        number_of_one = bin(state // 4).count('1')
        if number_of_one % 2 == 0 and number_of_one > 2:
            first_bin = bin((state // 4) // (2 ** CARD_NUM))
            second_bin = bin((state // 4) % (2 ** CARD_NUM))
            first_pos = [2 - first_bin.find('1'), CARD_NUM - first_bin[::-1].find('1') - 1]
            second_pos = [2 - second_bin.find('1'), CARD_NUM - second_bin[::-1].find('1') - 1]
            if -1 in first_pos + second_pos or first_bin.count('1') != second_bin.count('1'):
                continue
            if first_pos[0] >= second_pos[1] or first_pos[1] <= second_pos[0]:
                continue
            q_table[state] = [0] * second_bin.count('1')

def read_q_table(path):
    '''
    Read Q-table from the given file.
    Args:
        path (str): file path
    Returns:
        None
    '''
    try:
        with open(path, 'r', encoding='UTF-8') as q_table_file:
            while True:
                line = q_table_file.readline()
                if line == '':
                    break
                splited_line = line.rstrip().split()
                q_table[int(splited_line[0])] = [float(x) for x in splited_line[1:]]
    except FileNotFoundError:
        print('File not found. [Initialize]')
        initialize_q_table()

def write_q_table(path):
    '''
    Write Q-table to the given file.
    Args:
        path (str): file path
    Returns:
        None
    '''
    with open(path, 'w', encoding='UTF-8') as q_table_file:
        for q_line in q_table.items():
            q_table_file.write(f'{q_line[0]} ')
            for q_value in q_line[1]:
                q_table_file.write(f'{q_value} ')
            q_table_file.write('\n')

def softmax(q_values, temperature):
    '''
    Return softmax probability list with given Q-values.
    Args:
        q_values (list): List of Q values
        temperature (float): Temperature value of softmax algorithm
    Returns:
        probabilities (list): Probability values
    '''
    exp_values = [np.exp(q_value / temperature) for q_value in q_values]
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

def select_action(action_probabilities):
    '''
    Select action with given probability list.
    Args:
        action_probabilities (list): Probability values of each action
    Returns:
        selected_action (int): index of chosen action
    '''
    num_actions = len(action_probabilities)
    action_indices = np.arange(num_actions)
    selected_action = np.random.choice(action_indices, p=action_probabilities)
    return selected_action

def cal_state(numbers, oppo_numbers, is_first=True, odd_flag=False):
    '''
    Calculate state value.
    Args:
        numbers (list): Card number list
        oppo_numbers (list): Card number list of opponent
        is_first (bool): Indicates whether player is the first or not
        odd_flag (bool): Indicates whether number of opponent is odd or not
    Returns:
        state (int): Calculated state
    '''
    state = 0
    for number in numbers:
        state += 2 ** (number + 1)
    for number in oppo_numbers:
        state += 2 ** (number + 1 + CARD_NUM)
    state += int(is_first) * 2 + int(odd_flag)
    return state

def get_max_q(numbers, oppo_numbers, is_first=True):
    '''
    Find maximum value from given Q-table line.
    Args:
        numbers (list): Card number list
        oppo_numbers (list): Card number list of opponent
        is_first (bool): Indicates whether player is the first or not
    Returns:
        max_q_val (float): The maximum Q-value
    '''
    if is_first:
        max_q_val = max(q_table[cal_state(numbers, oppo_numbers)])
    else:
        max_q_val = max(
            q_table[cal_state(numbers, oppo_numbers, is_first, True)] +
            q_table[cal_state(numbers, oppo_numbers, is_first, False)]
        )
    return max_q_val

def q_choose_number(numbers, oppo_numbers, is_first=True, odd_flag=False, epsilon=0, soft_bound=0):
    '''
    Choose a number from number list using Q-table.
    Args:
        numbers (list): Card number list
        oppo_numbers (list): Card number list of opponent
        is_first (bool): Indicates whether player is the first or not
        odd_flag (bool): Indicates whether number of opponent is odd or not
        epsilon (float): A constant used to determine whether use Q-table or not (default: 0)
        soft_bound (int): Bound indicates whether use soft-max or max
    Returns:
        Tuple containing the following elements:
        action (int): index of chosen action
        state (int): present state
        action_q_value (float): Q-value of chosen action
    '''
    # calculate present state
    state = cal_state(numbers, oppo_numbers, is_first, odd_flag)

    # choose action with epsilon-greedy policy
    if np.random.random() < epsilon:
        action = np.random.randint(len(numbers))
    elif soft_bound > 0 and len(numbers) > soft_bound:
        action = select_action(softmax(q_table[state], TEMPERATURE))
    else:
        action = q_table[state].index(max(q_table[state]))
    action_q_value = q_table[state][action]
    return action, state, action_q_value

def q_learning(epochs, l_rate=0.1, d_factor=0.9, epsilon=0.1):
    '''
    Execute Q-learning.
    Args:
        epochs (int): Number of episode
        l_rate (float): Learning rate (default: 0.1)
        d_factor (float): Discount factor (default: 0.9)
        epsilon (float): A constant used to determine whether use Q-table or not (default: 0.1)
    Returns:
        None
    '''
    for episode in tqdm(range(epochs), mininterval=10):
        r_epsilon = epsilon / (1 + (episode / epochs))
        scores, rewards = [0] * PLAYER_NUM, [0] * PLAYER_NUM
        numbers = [list(range(1, CARD_NUM + 1)), list(range(1, CARD_NUM + 1))]
        first = int(np.random.randint(PLAYER_NUM))
        second = 1 - first
        states = [[] for _ in range(PLAYER_NUM)]
        game_over = False

        for round_num in range(CARD_NUM):
            first_action, first_state, first_q = q_choose_number(
                numbers[first],
                numbers[second],
                is_first=True,
                odd_flag=False,
                epsilon=r_epsilon
            )
            second_action, second_state, second_q = q_choose_number(
                numbers[second],
                numbers[first],
                is_first=False,
                odd_flag=numbers[first][first_action] % 2 != 0,
                epsilon=r_epsilon
            )
            first_number = numbers[first].pop(first_action)
            second_number = numbers[second].pop(second_action)

            for i in range(PLAYER_NUM):
                if numbers[i][0] >= numbers[1 - i][-1]:
                    scores[i] += CARD_NUM - round_num - 1
                    if numbers[i][0] == numbers[1 - i][-1]:
                        scores[i] -= 1
                    first_next_max_q, second_next_max_q = 0, 0
                    game_over = True
                    break

            if not game_over:
                first_next_max_q = get_max_q(numbers[first], numbers[second], is_first=False)
                second_next_max_q = get_max_q(numbers[second], numbers[first])

            states[first].append((
                first_state, first_action, l_rate * (d_factor * first_next_max_q - first_q)
            ))
            states[second].append((
                second_state, second_action, l_rate * (d_factor * second_next_max_q - second_q)
            ))

            if first_number > second_number:
                rewards[first], rewards[second] = ROUND_REWARD, -ROUND_REWARD
                scores[first] += 1
            elif first_number < second_number:
                rewards[first], rewards[second] = -ROUND_REWARD, ROUND_REWARD
                scores[second] += 1

            q_table[first_state][first_action] += l_rate * rewards[first] + states[first][-1][2]
            q_table[second_state][second_action] += l_rate * rewards[second] + states[second][-1][2]

            if game_over:
                break
            first, second = second, first

        scores[FIRST] += int(numbers[FIRST] > numbers[SECOND])
        scores[SECOND] += int(numbers[FIRST] < numbers[SECOND])
        if scores[FIRST] != scores[SECOND]:
            winner = FIRST if scores[FIRST] > scores[SECOND] else SECOND
            for coordinate in states[winner]:
                q_table[coordinate[0]][coordinate[1]] += l_rate * FINAL_REWARD + coordinate[2]
            for coordinate in states[1 - winner]:
                q_table[coordinate[0]][coordinate[1]] += -l_rate * FINAL_REWARD + coordinate[2]

q_table = {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q-learning example')
    parser.add_argument('--epochs', type=int, default=10000, help='number of learning episodes')
    parser.add_argument('--learning', type=float, default=0.1, help='learning rate')
    parser.add_argument('--discount', type=float, default=0.9, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon value')
    parser.add_argument('--input', type=str, default='', help='Q-table input file path')
    parser.add_argument('--output', type=str, default='q_table.txt',
                        help='Q-table output file path')
    parser.add_argument('-r', '--read', action='store_true', help='read Q-table from file option')
    args = parser.parse_args()
    if args.read:
        read_q_table(args.input)
    else:
        initialize_q_table()
    q_learning(args.epochs, args.learning, args.discount, args.epsilon)
    write_q_table(args.output)
