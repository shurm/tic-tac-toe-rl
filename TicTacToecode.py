import copy
import random


class TicTacToe:
    def __init__(self):
        self.board = [[0 for i in range(3)] for j in range(3)]
        self.current_turn = 0
        self.characters_to_place = {1: 'X', -1: 'O', 0: '-'}
        self.winning_combos = (
            [6, 7, 8], [3, 4, 5], [0, 1, 2], [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        )

    def get_state(self):
        return copy.deepcopy(self.board)

    def copy(self):
        copyTTT = TicTacToe()
        copyTTT.board = copy.deepcopy(self.board)
        copyTTT.current_turn = self.current_turn
        return copyTTT

    def get_winner_id(self):
        for combo in self.winning_combos:
            set_combo = set([self.board[position // 3][position % 3] for position in combo])
            if len(set_combo) == 1 and 0 not in set_combo:
                return set_combo.pop()
        return 0

    def game_over(self):
        return self.get_winner_id() != 0 or len(self.legal_moves()) == 0

    def get_reward(self):
        winner_id = self.get_winner_id()
        # someone won
        if winner_id != 0:
            return winner_id

        # draw
        if len(self.legal_moves()) == 0:
            return 0.5
        # game not over
        return -1

    def legal_moves(self):
        if self.get_winner_id() != 0:
            return []
        legal_moves = []
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == 0:
                    legal_moves.append(r * 3 + c)
        return legal_moves

    def perform_move(self, move):
        if move not in self.legal_moves():
            return False
        r = move // 3
        c = move % 3
        self.board[r][c] = -2 * self.current_turn + 1
        self.current_turn = (self.current_turn + 1) % 2
        return True

    def successors(self):
        legal_moves = self.legal_moves()
        successors = [None] * len(legal_moves)
        for i in range(len(legal_moves)):
            successors[i] = self.copy()
            successors[i].perform_move(legal_moves[i])
        return successors

    def int_to_symbol(self, i):
        return self.characters_to_place[i]

    def __str__(self):
        lines = ["".join(map(self.int_to_symbol, row)) for row in self.board]
        return "\n".join(lines)


model_table = ({}, None)
GAMMA = 0.9


def get_probs(board):
    successors = board.successors()
    probs = [0] * len(successors)
    for i in range(len(successors)):
        successor = successors[i]

        hash_string = str(successor)
        v_table,_ = model_table
        probs[i] = v_table.get(hash_string, 0)
        # else:
        # probs[i] = 0

    probs = [(-2*board.current_turn+1)*prob for prob in probs]
    minV = min(probs)
    maxV = max(probs)
    probs = [(prob - minV + 1) / ((maxV - minV) + len(probs)) for prob in probs]

    return probs

def update_v(history):
    board = history.pop()
    hash_string = str(board)
    v_table, _ = model_table
    v_table[hash_string] = board.get_reward()

    functions = [max, min]
    while len(history) > 0:
        board = history.pop()
        successors = board.successors()
        v_successor_values = []
        for successor in successors:
            hash_string = str(successor)
            if hash_string in v_table:
                v_successor_values.append(v_table[hash_string])
        desired_function = functions[board.current_turn]
        v = desired_function(v_successor_values)
        hash_string = str(board)
        v_table[hash_string] = GAMMA * v

for i in range(10000):
    board = TicTacToe()
    history = [board]
    while not board.game_over():
        successors = board.successors()
        probs = get_probs(board)
        chosen_successor = random.choices(successors, probs, k=1)[0]
        history.append(chosen_successor)
        board = chosen_successor
    #if board.get_winner_id()==-1:
    #    board.get_winner_id()
    update_v(history)

def get_best_successor_index(board):
    # print(len(successors))
    probs = get_probs(board)
    max_prob = max(probs)
    max_i = probs.index(max_prob)
    return max_i

# v_table = sorted(v_table.items(), key=lambda x: x[1], reverse=True)
# print(v_table)

board = TicTacToe()
while not board.game_over():
    print(board)
    print("")

    move = input("Enter move:")
    board.perform_move(int(move))

    successors = board.successors()
    i = get_best_successor_index(board)

    board = successors[i]
print(board)
