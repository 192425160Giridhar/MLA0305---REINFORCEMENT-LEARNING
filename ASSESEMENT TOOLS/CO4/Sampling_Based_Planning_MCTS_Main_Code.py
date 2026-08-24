# Sampling-Based Planning & Data Generation Using Reinforcement Learning
# Main source code - MCTS vs Greedy vs Random
# Run this file directly with Python or paste it into one Jupyter cell.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# ------------------------------------------------
GRID_SIZE = 8
START = (0, 0)
GOAL = (7, 7)

ACTIONS = {
    0: (-1, 0),   # Up
    1: (1, 0),    # Down
    2: (0, -1),   # Left
    3: (0, 1)     # Right
}

STEP_REWARD = -1
GOAL_REWARD = 100
COLLISION_PENALTY = -10
MAX_STEPS = 60
N_EPISODES = 50

# ------------------------------------------------
# 2. GENERATE ENVIRONMENT
# ------------------------------------------------
def generate_obstacles(seed=42, density=0.15):
    rng = np.random.default_rng(seed)
    obstacles = set()

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x, y) not in [START, GOAL]:
                if rng.random() < density:
                    obstacles.add((x, y))
    return obstacles

obstacles = generate_obstacles()

# ------------------------------------------------
# 3. GENERATE 1000-RECORD DATASET
# ------------------------------------------------
rng = np.random.default_rng(42)
records = []

for episode in range(1, 1001):
    x = rng.integers(0, GRID_SIZE)
    y = rng.integers(0, GRID_SIZE)

    distance = abs(x - GOAL[0]) + abs(y - GOAL[1])

    records.append({
        "Episode": episode,
        "State_X": x,
        "State_Y": y,
        "Goal_X": GOAL[0],
        "Goal_Y": GOAL[1],
        "Distance_to_Goal": distance,
        "Obstacle_Density": len(obstacles) / (GRID_SIZE ** 2),
        "Step_Cost": STEP_REWARD,
        "Goal_Reward": GOAL_REWARD,
        "Collision_Penalty": COLLISION_PENALTY
    })

dataset = pd.DataFrame(records)

# ------------------------------------------------
# 4. ENVIRONMENT FUNCTIONS
# ------------------------------------------------
def valid_state(state):
    x, y = state
    return (
        0 <= x < GRID_SIZE
        and 0 <= y < GRID_SIZE
        and state not in obstacles
    )

def next_state(state, action):
    dx, dy = ACTIONS[action]

    new_state = (
        state[0] + dx,
        state[1] + dy
    )

    if not valid_state(new_state):
        return state, COLLISION_PENALTY, False

    if new_state == GOAL:
        return new_state, GOAL_REWARD, True

    return new_state, STEP_REWARD, False

# ------------------------------------------------
# 5. MCTS NODE
# ------------------------------------------------
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(ACTIONS.keys())

    def fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration=1.414):
        scores = {}

        for action, child in self.children.items():
            if child.visits == 0:
                scores[action] = float("inf")
            else:
                exploitation = child.value / child.visits
                exploration_term = (
                    exploration *
                    np.sqrt(
                        np.log(self.visits + 1) / child.visits
                    )
                )
                scores[action] = exploitation + exploration_term

        return max(scores, key=scores.get)

# ------------------------------------------------
# 6. MONTE CARLO TREE SEARCH
# ------------------------------------------------
class MCTS:
    def __init__(self, simulations=100, depth=15):
        self.simulations = simulations
        self.depth = depth

    def rollout(self, state):
        current_state = state
        total_reward = 0

        for _ in range(self.depth):
            if current_state == GOAL:
                break

            action = np.random.choice(list(ACTIONS.keys()))

            new_state, reward, done = next_state(
                current_state, action
            )

            total_reward += reward
            current_state = new_state

            if done:
                break

        return total_reward

    def search(self, root_state):
        root = MCTSNode(root_state)

        for _ in range(self.simulations):
            node = root

            # Selection
            while node.fully_expanded() and node.children:
                action = node.best_child()
                node = node.children[action]

            # Expansion
            if node.untried_actions:
                action = node.untried_actions.pop()

                new_state, _, _ = next_state(
                    node.state, action
                )

                child = MCTSNode(
                    new_state,
                    parent=node,
                    action=action
                )

                node.children[action] = child
                node = child

            # Simulation
            reward = self.rollout(node.state)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        if not root.children:
            return np.random.choice(list(ACTIONS.keys()))

        return max(
            root.children,
            key=lambda a: root.children[a].visits
        )

# ------------------------------------------------
# 7. MCTS AGENT
# ------------------------------------------------
def run_mcts_episode(simulations=100):
    planner = MCTS(simulations=simulations, depth=15)

    state = START
    total_reward = 0
    path = [state]
    collisions = 0

    for _ in range(MAX_STEPS):
        if state == GOAL:
            break

        action = planner.search(state)

        new_state, reward, done = next_state(
            state, action
        )

        if (
            new_state == state
            and reward == COLLISION_PENALTY
        ):
            collisions += 1

        total_reward += reward
        state = new_state
        path.append(state)

        if done:
            break

    return {
        "Reward": total_reward,
        "Steps": len(path) - 1,
        "Success": int(state == GOAL),
        "Collisions": collisions,
        "Path": path
    }

# ------------------------------------------------
# 8. GREEDY AGENT
# ------------------------------------------------
def run_greedy_episode():
    state = START
    total_reward = 0
    path = [state]
    collisions = 0

    for _ in range(MAX_STEPS):
        if state == GOAL:
            break

        candidates = []

        for action in ACTIONS:
            new_state, _, done = next_state(state, action)

            if new_state != state or done:
                distance = (
                    abs(new_state[0] - GOAL[0])
                    + abs(new_state[1] - GOAL[1])
                )
                candidates.append((distance, action))

        if candidates:
            _, action = min(candidates)
        else:
            action = np.random.choice(list(ACTIONS.keys()))

        new_state, reward, done = next_state(state, action)

        if (
            new_state == state
            and reward == COLLISION_PENALTY
        ):
            collisions += 1

        total_reward += reward
        state = new_state
        path.append(state)

        if done:
            break

    return {
        "Reward": total_reward,
        "Steps": len(path) - 1,
        "Success": int(state == GOAL),
        "Collisions": collisions,
        "Path": path
    }

# ------------------------------------------------
# 9. RANDOM AGENT
# ------------------------------------------------
def run_random_episode():
    state = START
    total_reward = 0
    path = [state]
    collisions = 0

    for _ in range(MAX_STEPS):
        if state == GOAL:
            break

        action = np.random.choice(list(ACTIONS.keys()))

        new_state, reward, done = next_state(state, action)

        if (
            new_state == state
            and reward == COLLISION_PENALTY
        ):
            collisions += 1

        total_reward += reward
        state = new_state
        path.append(state)

        if done:
            break

    return {
        "Reward": total_reward,
        "Steps": len(path) - 1,
        "Success": int(state == GOAL),
        "Collisions": collisions,
        "Path": path
    }

# ------------------------------------------------
# 10. RUN EXPERIMENTS
# ------------------------------------------------
results = []

print("=" * 70)
print("RUNNING SAMPLING-BASED PLANNING EXPERIMENT")
print("=" * 70)

for episode in range(1, N_EPISODES + 1):
    mcts = run_mcts_episode(100)
    results.append({
        "Episode": episode,
        "Algorithm": "MCTS",
        "Reward": mcts["Reward"],
        "Steps": mcts["Steps"],
        "Success": mcts["Success"],
        "Collisions": mcts["Collisions"]
    })

    greedy = run_greedy_episode()
    results.append({
        "Episode": episode,
        "Algorithm": "Greedy",
        "Reward": greedy["Reward"],
        "Steps": greedy["Steps"],
        "Success": greedy["Success"],
        "Collisions": greedy["Collisions"]
    })

    random_agent = run_random_episode()
    results.append({
        "Episode": episode,
        "Algorithm": "Random",
        "Reward": random_agent["Reward"],
        "Steps": random_agent["Steps"],
        "Success": random_agent["Success"],
        "Collisions": random_agent["Collisions"]
    })

results_df = pd.DataFrame(results)

print("Experiment completed successfully.")
print("Total experiment records:", len(results_df))

# ------------------------------------------------
# 11. DATASET OUTPUT
# ------------------------------------------------
print("\n" + "=" * 70)
print("GENERATED DATASET")
print("=" * 70)
print(f"Records: {dataset.shape[0]} | Parameters: {dataset.shape[1]}")
display(dataset.head(10))

# ------------------------------------------------
# 12. SUMMARY TABLE
# ------------------------------------------------
summary = (
    results_df
    .groupby("Algorithm")
    .agg(
        Average_Reward=("Reward", "mean"),
        Reward_Std=("Reward", "std"),
        Success_Rate=("Success", "mean"),
        Average_Steps=("Steps", "mean"),
        Average_Collisions=("Collisions", "mean")
    )
    .reset_index()
)

summary["Success_Rate"] *= 100
summary = summary.round(2)

print("\n" + "=" * 70)
print("FINAL PERFORMANCE SUMMARY")
print("=" * 70)
display(summary)

# ------------------------------------------------
# 13. GRAPH 1 - LEARNING CURVE
# ------------------------------------------------
plt.figure(figsize=(11, 6))

for algorithm in results_df["Algorithm"].unique():
    temp = results_df[results_df["Algorithm"] == algorithm]
    rolling = temp["Reward"].rolling(5, min_periods=1).mean()

    plt.plot(
        temp["Episode"],
        rolling,
        linewidth=2,
        label=algorithm
    )

plt.title(
    "Learning Performance: Episode vs Average Reward",
    fontsize=15,
    fontweight="bold"
)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend(title="Algorithm")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 14. GRAPH 2 - REWARD COMPARISON
# ------------------------------------------------
reward_data = (
    results_df
    .groupby("Algorithm")["Reward"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))
bars = plt.bar(reward_data.index, reward_data.values)

plt.title(
    "Average Reward Comparison",
    fontsize=15,
    fontweight="bold"
)
plt.xlabel("Algorithm")
plt.ylabel("Average Reward")

for bar, value in zip(bars, reward_data.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 15. GRAPH 3 - SUCCESS RATE
# ------------------------------------------------
success_data = (
    results_df
    .groupby("Algorithm")["Success"]
    .mean() * 100
)

plt.figure(figsize=(9, 6))
bars = plt.bar(success_data.index, success_data.values)

plt.title(
    "Success Rate Comparison",
    fontsize=15,
    fontweight="bold"
)
plt.xlabel("Algorithm")
plt.ylabel("Success Rate (%)")
plt.ylim(0, 100)

for bar, value in zip(bars, success_data.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.1f}%",
        ha="center",
        fontweight="bold"
    )

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 16. GRAPH 4 - AVERAGE STEPS
# ------------------------------------------------
steps_data = (
    results_df
    .groupby("Algorithm")["Steps"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(9, 6))
bars = plt.bar(steps_data.index, steps_data.values)

plt.title(
    "Average Steps Required for Navigation",
    fontsize=15,
    fontweight="bold"
)
plt.xlabel("Algorithm")
plt.ylabel("Average Number of Steps")

for bar, value in zip(bars, steps_data.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 17. GRAPH 5 - REWARD DISTRIBUTION
# ------------------------------------------------
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=results_df,
    x="Algorithm",
    y="Reward"
)

plt.title(
    "Reward Distribution Comparison",
    fontsize=15,
    fontweight="bold"
)
plt.xlabel("Algorithm")
plt.ylabel("Episode Reward")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------------------------
# 18. GRAPH 6 - COLLISION COMPARISON
# ------------------------------------------------
collision_data = (
    results_df
    .groupby("Algorithm")["Collisions"]
    .mean()
)

plt.figure(figsize=(9, 6))
bars = plt.bar(collision_data.index, collision_data.values)

plt.title(
    "Average Collision Comparison",
    fontsize=15,
    fontweight="bold"
)
plt.xlabel("Algorithm")
plt.ylabel("Average Collisions")

for bar, value in zip(bars, collision_data.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 19. FINAL MCTS NAVIGATION PATH
# ------------------------------------------------
final_result = run_mcts_episode(500)

plt.figure(figsize=(8, 8))

plt.xlim(-0.5, GRID_SIZE - 0.5)
plt.ylim(-0.5, GRID_SIZE - 0.5)
plt.xticks(range(GRID_SIZE))
plt.yticks(range(GRID_SIZE))
plt.grid(True, linestyle="--", alpha=0.5)

for x, y in obstacles:
    plt.scatter(y, x, marker="s", s=700)

plt.scatter(
    START[1], START[0],
    marker="o",
    s=300,
    label="Start"
)

plt.scatter(
    GOAL[1], GOAL[0],
    marker="*",
    s=400,
    label="Goal"
)

path = final_result["Path"]
px = [p[1] for p in path]
py = [p[0] for p in path]

plt.plot(
    px, py,
    marker="o",
    linewidth=2,
    label="MCTS Path"
)

plt.title(
    "Final MCTS Planned Navigation Path",
    fontsize=15,
    fontweight="bold"
)
plt.xlabel("Grid Column")
plt.ylabel("Grid Row")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 20. FINAL EVALUATION
# ------------------------------------------------
best = (
    summary
    .sort_values("Average_Reward", ascending=False)
    .iloc[0]
)

print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)
print(f"Best Algorithm       : {best['Algorithm']}")
print(f"Average Reward       : {best['Average_Reward']:.2f}")
print(f"Success Rate         : {best['Success_Rate']:.2f}%")
print(f"Average Steps        : {best['Average_Steps']:.2f}")
print(f"Average Collisions   : {best['Average_Collisions']:.2f}")

print("\nFinal MCTS Episode")
print(f"Reward               : {final_result['Reward']}")
print(f"Steps                : {final_result['Steps']}")
print(f"Success              : {final_result['Success']}")
print(f"Collisions           : {final_result['Collisions']}")

print("\n" + "=" * 70)
print("IMPLEMENTATION COMPLETED SUCCESSFULLY")
print("=" * 70)
