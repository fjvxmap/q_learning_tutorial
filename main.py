'''Play game with trained computer'''
import train.train as trained

PLAYER, COMPUTER = 0, 1
NAMES = ['Player', 'Computer']

def choose_number(numbers, oppo_numbers, is_player=False, is_first = True, odd_flag = False):
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
        chosen_number (int): chosen number
        numbers (list): number list after choosing
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
        numbers.remove(chosen_number)
    else:
        _, _, chosen_number, numbers, _ = trained.q_choose_number(
            numbers,
            oppo_numbers,
            is_first,
            odd_flag,
            soft_bound=trained.CARD_NUM / 2
        )
    return chosen_number, numbers

def is_game_over(numbers, remaining_round):
    '''
    Check whether the game is over or not.
    Args:
        numbers (list): Card number lists of player and computer
        remaining_round (int): Number of remaining rounds
    Returns:
        Tuple containing the following elements:
        game_ended (bool): Indicates whether the game is over or not
        winner (int): Id of winner (if the game is not over, None)
    '''
    player_win, computer_win = 0, 0
    for i in range(remaining_round):
        if len([n for n in numbers[COMPUTER] if numbers[PLAYER][i] > n]) == remaining_round:
            player_win += 1
        if len([n for n in numbers[PLAYER] if numbers[COMPUTER][i] > n]) == remaining_round:
            computer_win += 1
    if player_win == remaining_round:
        game_ended, winner = True, PLAYER
    elif computer_win == remaining_round:
        game_ended, winner = True, COMPUTER
    else:
        game_ended, winner = False, None
    return game_ended, winner

def play_game():
    '''
    Play game with computer
    Args:
        None
    Returns:
        None
    '''
    scores = [0] * trained.PLAYER_NUM
    numbers = [list(range(1, trained.CARD_NUM + 1)), list(range(1, trained.CARD_NUM + 1))]
    first = int(trained.np.random.randint(trained.PLAYER_NUM))
    second = 1 - first

    for round_num in range(1, trained.CARD_NUM + 1):
        print(f'[Round {round_num}]')
        print("Computer's available numbers:\t", numbers[COMPUTER])
        print("Player's available numbers:\t", numbers[PLAYER])
        prev_numbers = numbers[first][:]
        first_number, numbers[first] = choose_number(
            numbers[first],
            numbers[second],
            first == PLAYER
        )
        if first_number % 2 == 0:
            odd_flag = 0
            print(f'{NAMES[first]} has chosen an even number.')
        else:
            odd_flag = 1
            print(f'{NAMES[first]} has chosen an odd number.')
        second_number, numbers[second] = choose_number(
            numbers[second],
            prev_numbers,
            second == PLAYER,
            is_first=False,
            odd_flag=odd_flag
        )
        print(f'Computer has chosen the number {first_number if first == COMPUTER else second_number}.')

        if first_number > second_number:
            print(f'{NAMES[first]} won!')
            scores[first] += 1
        elif first_number < second_number:
            print(f'{NAMES[second]} won!')
            scores[second] += 1
        else:
            print("It's a tie!")
        print()

        game_ended, winner = is_game_over(numbers, trained.CARD_NUM - round_num)
        if game_ended:
            scores[winner] += trained.CARD_NUM - round_num
            break

        first, second = second, first

    print("Game over!")
    print("Player's score:", scores[PLAYER])
    print("Computer's score:", scores[COMPUTER])

    if scores[PLAYER] > scores[COMPUTER]:
        print("Congratulations! You are the ultimate winner!")
    elif scores[PLAYER] < scores[COMPUTER]:
        print("Computer is the ultimate winner!")
    else:
        print("The final result is a tie!")


parser = trained.argparse.ArgumentParser(description='Play game with trained computer')
parser.add_argument('--input', type=str, default='q_table.txt', help='Q-table input file path')
args = parser.parse_args()

trained.read_q_table(args.input)
play_game()
