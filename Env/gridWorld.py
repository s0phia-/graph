import random
import numpy as np
from math import prod
import copy


grid = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        # [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        # [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        # [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

win_state = [8, 8]


class GridWorld:
    def __init__(self, random_actions=0, win_state=win_state, layout=grid):
        """
        :param random_actions: probability of the env ignoring the chosen action and selecting a move at random
        :param win_state: coordinates of goal state
        :param layout: 0,1 depiction of grid, where 1s are edges
        """
        self.grid = np.array(layout)
        self.grid_shape = self.grid.shape
        self.num_rows = self.grid_shape[0]
        self.num_cols = self.grid_shape[1]
        self.num_states = self.num_rows * self.num_cols
        self.win_state = win_state
        if random_actions is False:
            self.prob_random_action = 0
        else:
            self.prob_random_action = random_actions
        self.state = self.get_random_state()
        # only calculate the state transition graph if it is explicitly called, as it is a long calculation
        self.__state_transition_graph = None
        self.__adjacency_matrix = None
        self.__valency_matrix = None

    @property
    def state_transition_graph(self):
        """
        if state transition graph attribute is called, calculate it
        """
        if self.__state_transition_graph is None:
            self.__state_transition_graph = self.calc_state_transition_graph()
            return self.__state_transition_graph

    @property
    def adjacency_matrix(self):
        if self.__adjacency_matrix is None:
            self.__adjacency_matrix = self.calc_adjacency_matrix()
            return self.__adjacency_matrix

    @property
    def valency_matrix(self):
        if self.__valency_matrix is None:
            self.__valency_matrix = self.calc_valency_matrix()
            return self.__valency_matrix

    def get_random_state(self):
        """
        :return: a random state which doesn't land on any of the grid edges (1s)
        """
        valid_state = False
        assert np.sum(self.grid) != prod(self.grid_shape)
        while not valid_state:
            state = random.randrange(self.grid_shape[0]), random.randrange(self.grid_shape[1])
            if self.grid[state] == 0:
                valid_state = True
        return state

    def step(self, action):
        """
        :param action: should be 0 (left), 1 (right), 2 (up), 3 (down)
        :return: state action led to
        """
        h, v = self.state
        if random.random() < self.prob_random_action:
            action = random.choice([0, 1, 2, 3])
        if action == 0:  # left
            state = h-1, v
        if action == 1:  # right
            state = h+1, v
        if action == 2:  # up
            state = h, v-1
        if action == 3:  # down
            state = h, v+1
        if self.grid[state] == 1:  # if new state is a wall, go to the old state
            state = h, v
        self.state = state
        done = (state == win_state)
        reward = done*10 - (1-done)  # reward is 10 if done, otherwise -1
        return state, reward, done, ""

    def render(self):
        """
        print text version of gridworld, with 2 indicating the present state
        """
        grid_with_state = copy.copy(self.grid)
        grid_with_state[self.state] = 2
        print(grid_with_state)

    def calc_adjacency_matrix(self):
        """
        :return: An |S| x |S| matrix with entry 1 if the states are adjacent, 0 otherwise
        """
        adjacency_matrix_long = np.zeros((self.num_rows, self.num_cols, self.num_rows, self.num_cols))

        for s0, s1 in np.ndindex(self.grid_shape):
            if self.grid[s0, s1] == 1:  # if the state is a wall
                adjacency_matrix_long[s0, s1, :, :] = -1  # mark as negative so this state can be removed later
                adjacency_matrix_long[:, :, s0, s1] = -1
                continue

            for i in [[0, 1], [1, 0], [0, -1], [-1, 0]]:  # all possible actions
                # new state following the action
                s_prime = [s0 + i[0], s1 + i[1]]
                s0_prime, s1_prime = s_prime

                # whether the action was horizontal or vertical
                change_axis = np.argmax(np.abs(i))

                # if the new state is out of bounds of the grid
                if s_prime[change_axis] < 0 or s_prime[change_axis] > self.grid_shape[change_axis]:
                    # add self loop and skip
                    #adjacency_matrix_long[s0, s1, s0, s1] = 1
                    continue

                if self.grid[s0_prime, s1_prime] == 1:  # if the new state is a wall
                    # add self loop and skip
                    #adjacency_matrix_long[s0, s1, s0, s1] = 1
                    adjacency_matrix_long[s0, s1, s0_prime, s1_prime] = -1
                    continue

                else:
                    # states are adjacent! Fill in with a 1
                    adjacency_matrix_long[s0, s1, s0_prime, s1_prime] = 1

        # reshape into |S| x |S|
        adjacency_matrix = adjacency_matrix_long.reshape((self.num_rows*self.num_cols,
                                                          self.num_rows*self.num_cols))
        adjacency_matrix_no_walls = self.remove_walls(adjacency_matrix)
        return adjacency_matrix_no_walls

    @staticmethod
    def remove_walls(matrix):
        wall_states = np.isin(matrix, -1)
        clean_matrix = matrix[~wall_states]
        clean_matrix_dim = int(len(clean_matrix) ** .5)
        clean_sq_matrix = clean_matrix.reshape((clean_matrix_dim, clean_matrix_dim))
        return clean_sq_matrix

    def calc_valency_matrix(self):
        vv = np.zeros(self.__adjacency_matrix.shape)
        for i in range(vv.shape[0]):
            # count number of states adjacent to state i
            vv[i][i] = min(np.sum(self.__adjacency_matrix[i]),1)
        return vv

    def calc_state_transition_graph(self):
        """
        calculate the state transition graph manually from the rules of the step function
        :return: state transition matrix, size d1 x d2 x d1 x d2 x |a| where di is a dimension of the gridworld
        """
        st_graph = np.zeros([self.grid_shape[0], self.grid_shape[1], self.grid_shape[0], self.grid_shape[1], 4])
        # probability of action, given randomness
        p = (1-self.prob_random_action) + self.prob_random_action/4
        # probability of an action being chosen at random
        randp = self.prob_random_action/4

        # iterate through every state
        for s0, s1 in np.ndindex(self.grid_shape):
            if self.grid[s0, s1] == 1:  # if the state is on a wall
                st_graph[s0, s1, :, :, :] = -1  # mark as negative (probabilities can't be negative)
                st_graph[:, :, s0, s1, :] = -1
                continue

            def state_action_transition_prob(self, st_graph, s0, s1, sign, randp, p, action):
                """
                fill out the state transition graph for a single state action pair
                """
                var = s0, s1
                s0_prime, s1_prime = s0 + sign[0], s1 + sign[1]
                if sum(sign) == 1:  # if the change is positive
                    dim = np.argmax(abs(np.array(sign)))  # figure out which dimension the change is in
                    condition = var[np.argmax(sign)] + 1 < self.grid_shape[dim]  # check the change is in grid limits
                if sum(sign) == -1:  # if the change is negative
                    condition = var[np.argmin(sign)] - 1 >= 0  # check the change is within grid limits
                if condition:
                    if self.grid[s0_prime, s1_prime] == 0:  # as long as the move is valid
                        st_graph[s0, s1, s0_prime, s1_prime, :] = randp
                        st_graph[s0, s1, s0_prime, s1_prime, action] = p
                    else:  # the move would take you into a wall
                        st_graph[s0, s1, s0, s1, action] = p  # so you stay in the same place instead
                        st_graph[s0, s1, s0_prime, s1_prime] = -1  # mark moving to a wall as -1 (probs can't be -ve)
                return st_graph

            # left
            st_graph = state_action_transition_prob(self, st_graph, s0, s1, [-1, 0], randp=randp, p=p, action=0)

            # right
            st_graph = state_action_transition_prob(self, st_graph, s0, s1, [+1, 0], randp=randp, p=p, action=1)

            # up
            st_graph = state_action_transition_prob(self, st_graph, s0, s1, [0, -1], randp=randp, p=p, action=2)

            # down
            st_graph = state_action_transition_prob(self, st_graph, s0, s1, [0, +1], randp=randp, p=p, action=3)

        return st_graph
