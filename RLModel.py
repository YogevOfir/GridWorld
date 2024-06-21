import random
import numpy as np
import math

# Constants
STEP = -0.04  # constant reward for non-terminal states
DISCOUNT = 0.5
NUM_ACTIONS = 4
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # Down, Left, Up, Right
W = 4
H = 3
REWARDS = [(1, 1, 0), (3, 2, 1), (3, 1, -1)]

# Parameters for Boltzmann exploration
TEMPERATURE = 1
TEMPERATURE_DECAY = 0.99

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Initializing rewards
def initial_rewards(W, H, rewards):
    R_grid = [[STEP for _ in range(W)] for _ in range(H)]
    for x, y, val in rewards:
        R_grid[y][x] = val
    return R_grid


# Boltzmann exploration policy
def boltzmann_exploration(Q, state, temperature):
    q_values = Q[state[0]][state[1]]
    exp_q = np.exp(np.array(q_values) / temperature)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(range(NUM_ACTIONS), p=probs)

# Get the next state based on the current state and action
def get_next_state(state, action, rewards):
    r, c = state
    dr, dc = ACTIONS[action]
    newR, newC = r + dr, c + dc
    if newR < 0 or newC < 0 or newR >= H or newC >= W or ((newR, newC) in [(py, px) for px, py, val in rewards if val == 0]):
        return (r, c)
    else:
        return (newR, newC)

# Learn MDP from experiences
def learn_mdp_from_experience(experience, W, H):
    T = np.zeros((H, W, NUM_ACTIONS, H, W))
    R = np.zeros((H, W, NUM_ACTIONS))
    N = np.zeros((H, W, NUM_ACTIONS))
    
    for (i, a, r, j) in experience:
        T[i[0], i[1], a, j[0], j[1]] += 1
        R[i[0], i[1], a] += r
        N[i[0], i[1], a] += 1

    for r in range(H):
        for c in range(W):
            for a in range(NUM_ACTIONS):
                if N[r, c, a] > 0:
                    R[r, c, a] /= N[r, c, a]
                    T[r, c, a] /= N[r, c, a]
    
    return T, R

# Solve MDP to obtain policy
def value_iteration(T, R, DISCOUNT, threshold=1e-4):
    V = np.zeros((H, W))
    policy = np.zeros((H, W), dtype=int)
    while True:
        delta = 0
        for r in range(H):
            for c in range(W):
                v = V[r, c]
                V[r, c] = max(sum(T[r, c, a, r2, c2] * (R[r, c, a] + DISCOUNT * V[r2, c2])
                                   for r2 in range(H) for c2 in range(W)) for a in range(NUM_ACTIONS))
                policy[r, c] = np.argmax([sum(T[r, c, a, r2, c2] * (R[r, c, a] + DISCOUNT * V[r2, c2])
                                              for r2 in range(H) for c2 in range(W)) for a in range(NUM_ACTIONS)])
                delta = max(delta, abs(v - V[r, c]))
        if delta < threshold:
            break
    return policy, V

# Iterative policy learning algorithm with Boltzmann exploration
def iterative_policy_learning(W, H, rewards, Q):
    # Q = [[[0 for _ in range(NUM_ACTIONS)] for _ in range(W)] for _ in range(H)]
    experience = []
    temperature = TEMPERATURE
    k = 0

    while True:
        k += 1
        state = (random.randint(0, H - 1), random.randint(0, W - 1))
        while state in [(py, px) for px, py, val in rewards if val != STEP]:
            state = (random.randint(0, H - 1), random.randint(0, W - 1))

        for _ in range(100):  # Choose a suitable number of steps for each episode
            action = boltzmann_exploration(Q, state, temperature)
            next_state = get_next_state(state, action, rewards)
            reward = R_grid[next_state[0]][next_state[1]]
            experience.append((state, action, reward, next_state))
            state = next_state

        T, R_mdp = learn_mdp_from_experience(experience, W, H)
        new_policy, _ = value_iteration(T, R_mdp, DISCOUNT)
        policy_stable = True

        # Update Q-values and check if the policy is stable
        for r in range(H):
            for c in range(W):
                for a in range(NUM_ACTIONS):
                    q_value = sum(T[r, c, a, r2, c2] * (R_mdp[r, c, a] + DISCOUNT * Q[r2][c2][np.argmax(Q[r2][c2])])
                                  for r2 in range(H) for c2 in range(W))
                    if abs(Q[r][c][a] - q_value) > 1e-4:
                        policy_stable = False
                    Q[r][c][a] = q_value

        if policy_stable:
            break
        temperature *= TEMPERATURE_DECAY

    return new_policy

# Print environment
def print_environment(arr, rewards, policy=False):
    rewards_dict = {(x, y): val for x, y, val in rewards}
    res = ""
    for r in range(H):
        res += "|"
        for c in range(W):
            if ((r, c) in [(py, px) for px, py, val in rewards if val == 0]):
                val = " WALL"
            elif ((r, c) in [(py, px) for px, py, val in rewards]):
                val = "  +" if rewards_dict[(c, r)] > 0 else "  -"
            else:
                if policy:
                    val = ["  v", "  <", "  ^", "  >"][arr[r][c]] 
                else:
                    val = f"{arr[r][c]}"
            res += " " + val[:5].ljust(5) + " |"  # format
        res += "\n"
    print(res)

# Main function
# Main function
# Main function
if __name__ == "__main__":
    rewards = [(x, H - 1 - y, val) for x, y, val in REWARDS]
    R_grid = initial_rewards(W, H, rewards)

    # Initialize Q-values
    Q = [[[0 for _ in range(NUM_ACTIONS)] for _ in range(W)] for _ in range(H)]

    # Run iterative policy learning
    policy = iterative_policy_learning(W, H, rewards, Q)

    # Calculate values (max Q-values)
    values = np.max(Q, axis=2)

    # Print values after training
    print("Values after training:\n")
    print_environment(values, rewards)

    # Print policy
    print("The optimal policy is:\n")
    print_environment(policy, rewards, True)

