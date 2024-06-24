import random
import copy

# Constants
STEP = -1  # constant reward for non-terminal states
DISCOUNT = 0.5
NUM_ACTIONS = 4
# P = 0.8  # probability for chosen action
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # Down, Left, Up, Right
W = 12
H = 4
REWARDS = [(1,0,-100),(2,0,-100),(3,0,-100),(4,0,-100),(5,0,-100),(6,0,-100),(7,0,-100),(8,0,-100),(9,0,-100),(10,0,-100),(11,0,1)]

# Parameters for Q-learning
LEARNING_RATE = 0.01
EPSILON = 0.01
EPSILON_DECAY = 0.995
EPISODES = 100000

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Initializing rewards
def initial_rewards(W, H, rewards):
    R = [[STEP for _ in range(W)] for _ in range(H)]
    for x, y, val in rewards:
        R[y][x] = val
    return R

# Epsilon-greedy policy
def epsilon_greedy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:  # Explore with probability epsilon
        return random.randint(0, NUM_ACTIONS - 1)
    else:  # Exploit with probability 1 - epsilon
        return max(range(NUM_ACTIONS), key=lambda a: Q[state[0]][state[1]][a])

# Get the next state based on the current state and action
def get_next_state(state, action, rewards):
    r, c = state
    dr, dc = ACTIONS[action]
    newR, newC = r + dr, c + dc
    if newR < 0 or newC < 0 or newR >= H or newC >= W or ((newR, newC) in [(py, px) for px, py, val in rewards if val == 0]):
        return (r, c)
    else:
        return (newR, newC)

# Q-learning algorithm
def q_learning(Q, rewards):
    epsilon = EPSILON
    epsilon_decay = 0.995  # Decay rate for epsilon
    for episode in range(EPISODES):
        state = (random.randint(0, H-1), random.randint(0, W-1))
        while state in [(py, px) for px, py, val in rewards if val != STEP]:
            state = (random.randint(0, H-1), random.randint(0, W-1))
        while state not in [(py, px) for px, py, val in rewards if val != STEP]:
            action = epsilon_greedy(Q, state, epsilon)  # Pass epsilon
            next_state = get_next_state(state, action, rewards)
            reward = R[next_state[0]][next_state[1]] 
            best_next_action = max(Q[next_state[0]][next_state[1]])
            Q[state[0]][state[1]][action] += LEARNING_RATE * (reward + DISCOUNT * best_next_action - Q[state[0]][state[1]][action])
            state = next_state
        epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon but ensure it doesn't drop below 0.01
    return Q

# Extract policy from Q-values
def extract_policy(Q):
    policy = [[0 for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            policy[r][c] = max(range(NUM_ACTIONS), key=lambda a: Q[r][c][a])
    return policy

# Extract values from Q-values
def extract_values(Q):
    values = [[0 for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            values[r][c] = max(Q[r][c])
    return values

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
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)

# Main function
if __name__ == "__main__":
    rewards = [(x, H-1-y, val) for x, y, val in REWARDS]
    R = initial_rewards(W, H, rewards)
    Q = [[[0 for _ in range(NUM_ACTIONS)] for _ in range(W)] for _ in range(H)]

    Q = q_learning(Q, rewards)
    
    values = extract_values(Q)
    
    print("Values after training:\n")
    print_environment(values, rewards)
    
    policy = extract_policy(Q)

    print("The optimal policy is:\n")
    print_environment(policy, rewards, True)
