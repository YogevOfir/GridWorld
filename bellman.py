import copy

STEP = -0.04 # constant reward for non-terminal states
DISCOUNT = 0.5
MAX_ERROR = 0.0001
NUM_ACTIONS = 4
P = 0.8  # probability for chosen action
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)] # Down, Left, Up, Right
W = 4
H = 3
REWARDS = [(1, 1, 0), (3, 2, 1), (3, 1, -1)]

def initial_mdp(W,H,rewards):
    V = [[0 for _ in range(W)] for _ in range(H)]
    for x, y, val in rewards:
        V[y][x] = val
    return V 

def printEnvironment(arr, rewards, policy=False):
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
                    val = str(arr[r][c])
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)
    
def get_utility(V, r, c, a, rewards):
    dr, dc = ACTIONS[a]
    newR, newC = r+dr, c+dc
    # if newR or newC is out of bounds or the new state is a wall, or the new state is a WALL (newR, newC) in REWARDS and val == 0
    if newR < 0 or newC < 0 or newR >= H or newC >= W or ((newR, newC) in [(py, px) for px, py, val in rewards if val == 0]):
        return V[r][c]
    else:
        return V[newR][newC]
        
        
def calculate_utility(V, r, c, a):
    utility = STEP
    utility += (1-P)/2 * DISCOUNT * get_utility(V, r, c, (a-1)%4, rewards)
    utility += P * DISCOUNT * get_utility(V, r, c, a, rewards)
    utility += (1-P)/2 * DISCOUNT * get_utility(V, r, c, (a+1)%4, rewards)
    return utility


def value_iteration(V, rewards):
    print("During the value iteration:\n")
    count = 0
    while True:
        nextV = copy.deepcopy(V)
        error = 0
        for r in range(H):
            for c in range(W):
                if (r,c) in [(py,px) for px,py,val in rewards]:
                    continue
                nextV[r][c] = max([calculate_utility(V, r, c, a) for a in range(NUM_ACTIONS)]) # Bellman update
                error = max(error, abs(nextV[r][c]-V[r][c]))
        V = nextV
        printEnvironment(V, rewards)
        if error < MAX_ERROR:
            print("Converged after " + str(count) + " iterations")
            break
        count += 1
    return V


def get_policy(V, rewards):
    policy = [[0 for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if (r,c) in [(py,px) for px,py,val in rewards]:
                continue
            
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculate_utility(V, r, c, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy
    
    
    
def upside_down(rewards):
    return [(x, H-1-y, val) for x, y, val in rewards]


rewards = upside_down(REWARDS)
# rewards = REWARDS
V = initial_mdp(W,H,rewards)
# V = initial_mdp(W,H,REWARDS)
# V = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
# V = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 0, 1]]
# print("Reward:" + str(REWARDS))
# print("Reverse Reward:" + str(upside_down(REWARDS)))
printEnvironment(V, rewards)

V = value_iteration(V, rewards)

policy = get_policy(V, rewards)

print("The optimal policy is:\n")
printEnvironment(policy, rewards, True)