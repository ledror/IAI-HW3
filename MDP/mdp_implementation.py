from copy import deepcopy
import numpy as np
from mdp import MDP

def best_action_utility(mdp: MDP, state, U) -> tuple:
    actions = list(mdp.transition_function.keys())

    best_action = None
    best_action_utility = float('-inf')

    for action in actions:
        action_utility = 0
        for i, conditional_prob in enumerate(mdp.transition_function[action]):
            next_state = mdp.step(state, actions[i])
            action_utility += conditional_prob * U[next_state[0]][next_state[1]]
        if action_utility > best_action_utility:
            best_action_utility = action_utility
            best_action = action

    return best_action, best_action_utility


def value_iteration(mdp: MDP, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    threshold = epsilon * (1 - mdp.gamma) / mdp.gamma
    U_prime = deepcopy(U_init)

    max_iteration = 10000
    iteration = 0

    while True:
        if iteration == max_iteration:
            break
        U = deepcopy(U_prime)
        delta = 0
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                reward = mdp.board[row][col]

                if reward == "WALL":
                    U_prime[row][col] = None
                    continue

                if (row, col) in mdp.terminal_states:
                    U_prime[row][col] = float(reward)
                else:
                    _, utility = best_action_utility(mdp, (row, col), U)
                    U_prime[row][col] = float(reward) + mdp.gamma * utility

                delta = max(delta, abs(U_prime[row][col] - U[row][col]))

        if delta < threshold:
            break

        iteration += 1

    return U



def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    pi = np.full((mdp.num_row, mdp.num_col), None)

    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            if mdp.board[row][col] == "WALL" or (row, col) in mdp.terminal_states:
                continue
            
            best_action, _ = best_action_utility(mdp, (row, col), U)
            pi[row][col] = best_action

    return pi

    


def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    
    actions = list(mdp.transition_function.keys())

    num_row = mdp.num_row
    num_col = mdp.num_col
    num_states = num_row * num_col

    prob_mat = np.zeros((num_states, num_states))
    reward_vec = np.zeros(num_states)

    for row in range(num_row):
        for col in range(num_col):
            reward = mdp.board[row][col]
            if reward == "WALL":
                continue

            reward_vec[row * num_col + col] = float(reward)

            if (row, col) in mdp.terminal_states:
                continue

            action = policy[row][col]
            for i, conditional_prob in enumerate(mdp.transition_function[action]):
                next_state = mdp.step((row, col), actions[i])
                prob_mat[row * num_col + col][next_state[0] * num_col + next_state[1]] += conditional_prob

    U = np.linalg.solve(np.eye(num_states) - mdp.gamma * prob_mat, reward_vec)
    U = U.reshape((num_row, num_col))

    # WALL states should have None as the utility
    for row in range(num_row):
        for col in range(num_col):
            if mdp.board[row][col] == "WALL":
                U[row][col] = None

    return U


def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    policy = deepcopy(policy_init)

    while True:
        U = policy_evaluation(mdp, policy)
        unchanged = True

        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                if mdp.board[row][col] == "WALL" or (row, col) in mdp.terminal_states:
                    continue

                best_action, _ = best_action_utility(mdp, (row, col), U)
                if best_action != policy[row][col]:
                    policy[row][col] = best_action
                    unchanged = False

        if unchanged:
            break

    return policy