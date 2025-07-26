import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TicTacToeEnv(gym.Env):
    """
    RL-friendly TicTacToe Environment.
    Board: 3x3 grid.
    State: 0 = empty, 1 = X (agent), 2 = O (opponent/random or second agent).
    Actions: 0-8 integer (row-major order).
    Reward: +1 win, -1 loss, 0 draw/ongoing/illegal.
    Episode ends on win/loss/draw.
    """

    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.action_space = spaces.Discrete(9)          # 9 cells, indexed 0-8
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(3, 3), dtype=np.int8)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1  # 1 (agent) always starts, 2 is opponent
        self.done = False
        return self.board.copy(), {}

    def step(self, action):
        row, col = divmod(action, 3)
        reward = 0.0
        info = {}

        # Invalid move
        if self.done or self.board[row, col] != 0:
            self.done = True
            reward = -1.0
            info['invalid_move'] = True
            return self.board.copy(), reward, self.done, False, info

        # Apply move
        self.board[row, col] = self.current_player

        # Check win/draw
        if self._is_win(self.current_player):
            self.done = True
            reward = 1.0
            return self.board.copy(), reward, self.done, False, info
        elif (self.board != 0).all():
            self.done = True
            reward = 0.0  # Draw
            return self.board.copy(), reward, self.done, False, info

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        # Opponent (random policy) if agent just played
        if self.current_player == 2:
            empty = np.argwhere(self.board == 0)
            if len(empty):
                idx = np.random.choice(len(empty))
                opp_row, opp_col = empty[idx]
                self.board[opp_row, opp_col] = 2
                if self._is_win(2):
                    self.done = True
                    reward = -1.0  # Agent lost
                    return self.board.copy(), reward, self.done, False, info
                elif (self.board != 0).all():
                    self.done = True
                    reward = 0.0  # Draw
                    return self.board.copy(), reward, self.done, False, info
            self.current_player = 1

        # Not done yet
        return self.board.copy(), reward, self.done, False, info

    def render(self, mode='human'):
        chars = {0: ".", 1: "X", 2: "O"}
        for row in self.board:
            print(" ".join([chars[v] for v in row]))
        print()

    def _is_win(self, player):
        b = self.board
        # Rows, columns, diagonals
        return any([
            np.all(b[i, :] == player) for i in range(3)
        ]) or any([
            np.all(b[:, j] == player) for j in range(3)
        ]) or (
            np.all(np.diag(b) == player) or np.all(
                np.diag(np.fliplr(b)) == player)
        )
