
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple
import json

# =========================
# Grid / Game parameters
# =========================
FIELD_WIDTH = 10
FIELD_HEIGHT = 10
NUM_HUNTERS = 3
MAX_TURNS_DEFAULT = 50
SIGHT_RADIUS = 3            # detection radius for hunters/prey features
SMELL_COOLDOWN_TURNS = 5    # turns until prey can smell again
FLARE_MEMORY_LIMIT = 3      # we expose up to 3 flares in the state

# =========================
# Entities
# =========================

@dataclass
class Prey:
    position: np.ndarray
    movement_speed: int = 1
    smell_cd: int = 0
    flares: List[np.ndarray] = None        # flares placed this turn
    seen_flares: List[np.ndarray] = None   # flares "remembered" (for features)
    current_move: str = "still"

    def __post_init__(self):
        if self.flares is None: self.flares = []
        if self.seen_flares is None: self.seen_flares = []

    def get_position(self):
        return self.position

    def _move_vec(self, action_name: str) -> np.ndarray:
        if action_name == "up":    return np.array([0, -1])
        if action_name == "down":  return np.array([0,  1])
        if action_name == "left":  return np.array([-1, 0])
        if action_name == "right": return np.array([ 1, 0])
        return np.array([0, 0])

    def move(self, action_name: str, hunters: List["Hunter"]):
        self.current_move = action_name
        if action_name == "smell":
            if self.smell_cd == 0:
                # Drop a flare at current position
                self.flares.append(self.position.copy())
                self.smell_cd = SMELL_COOLDOWN_TURNS
            # no spatial move
            return

        # spatial move
        delta = self._move_vec(action_name)
        # note: our position array is [x, y]; clamp in bounds
        nx = int(np.clip(self.position[0] + delta[0], 0, FIELD_WIDTH - 1))
        ny = int(np.clip(self.position[1] + delta[1], 0, FIELD_HEIGHT - 1))
        self.position = np.array([nx, ny], dtype=int)

    def decay_cooldowns(self):
        if self.smell_cd > 0:
            self.smell_cd -= 1


@dataclass
class Hunter:
    position: np.ndarray
    detected: bool = False  # whether the hunter is currently "seen" by prey (for features)
    aware: bool = False     # whether hunter is actively tracking prey (after flare etc.)

    def _step_toward(self, target_xy: np.ndarray):
        # greedy Manhattan move (one step) toward target
        dx = target_xy[0] - self.position[0]
        dy = target_xy[1] - self.position[1]
        if abs(dx) > abs(dy):
            step = np.array([np.sign(dx), 0], dtype=int)
        elif abs(dy) > 0:
            step = np.array([0, np.sign(dy)], dtype=int)
        else:
            step = np.array([0, 0], dtype=int)
        nx = int(np.clip(self.position[0] + step[0], 0, FIELD_WIDTH - 1))
        ny = int(np.clip(self.position[1] + step[1], 0, FIELD_HEIGHT - 1))
        self.position = np.array([nx, ny], dtype=int)

    def move(self, prey_xy: np.ndarray, hunters: List["Hunter"], prey: Prey):
        # simple policy: if aware or detected -> chase; else random walk
        if self.aware or self.detected:
            self._step_toward(prey_xy)
        else:
            # small random patrol
            r = random.random()
            if r < 0.25:   step = np.array([ 1, 0])
            elif r < 0.50: step = np.array([-1, 0])
            elif r < 0.75: step = np.array([ 0, 1])
            else:          step = np.array([ 0,-1])
            nx = int(np.clip(self.position[0] + step[0], 0, FIELD_WIDTH - 1))
            ny = int(np.clip(self.position[1] + step[1], 0, FIELD_HEIGHT - 1))
            self.position = np.array([nx, ny], dtype=int)


# =========================
# Spawning & Utilities
# =========================

def spawning(field_height: int = FIELD_HEIGHT, field_width: int = FIELD_WIDTH) -> Tuple[Prey, List[Hunter]]:
    # sample distinct positions
    occupied = set()
    def rand_cell():
        while True:
            x = random.randint(0, field_width - 1)
            y = random.randint(0, field_height - 1)
            if (x, y) not in occupied:
                occupied.add((x, y))
                return np.array([x, y], dtype=int)

    prey = Prey(position=rand_cell(), movement_speed=1)
    hunters = [Hunter(position=rand_cell()) for _ in range(NUM_HUNTERS)]
    # initial detect flags based on distance
    update_detection(prey, hunters)
    return prey, hunters

def update_detection(prey: Prey, hunters: List[Hunter]):
    p = prey.position
    for h in hunters:
        d = np.linalg.norm(p - h.position)
        h.detected = d <= SIGHT_RADIUS
        # awareness may persist via flares; here we keep it as-is unless set elsewhere


# =========================
# DQN Agent
# =========================

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.0001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = gamma   
        self.epsilon = epsilon  
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DQN(state_size, 256, action_size).to(self.device)
        self.target_network = DQN(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_valid_actions(self, prey: Prey, field_width: int, field_height: int):
        x, y = prey.position
        valid = []
        if y > 0: valid.append(0)                   # up
        if y < field_height - 1: valid.append(1)    # down
        if x > 0: valid.append(2)                   # left
        if x < field_width - 1: valid.append(3)     # right
        valid.append(4)                              # still
        if prey.smell_cd == 0: valid.append(5)      # smell
        return valid 

    def act(self, state, prey: Prey, field_width: int, field_height: int):
        if np.random.random() <= self.epsilon:
            valid_actions = self.get_valid_actions(prey, field_width, field_height)
            return random.choice(valid_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor).detach().cpu().numpy()[0]
        valid_actions = self.get_valid_actions(prey, field_width, field_height)
        invalid_actions = [a for a in range(self.action_size) if a not in valid_actions]
        q_values[invalid_actions] = -float("inf")
        return int(np.argmax(q_values))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =========================
# State & Reward
# =========================

def get_state_representation(prey: Prey, hunters: List[Hunter], field_width: int, field_height: int, remaining_moves: int):
    state = []
    prey_x, prey_y = prey.position
    # prey pos (normalized)
    state.extend([prey_x / field_width, prey_y / field_height])

    # up to 3 flares (normalized); only those seen are filled, others [0,0]
    flare_count = 0
    for flare in prey.flares:
        if any(np.array_equal(flare, seen) for seen in prey.seen_flares):
            state.extend([flare[0] / field_width, flare[1] / field_height])
            flare_count += 1
        else:
            state.extend([0, 0])
            flare_count += 1
    while flare_count != FLARE_MEMORY_LIMIT:
        state.extend([0, 0])
        flare_count += 1

    # fraction of visible hunters
    visible_hunters = sum(1 for h in hunters if h.detected)
    visible_count = visible_hunters / max(1, len(hunters))
    state.append(visible_count)

    # remaining moves (normalize by 2 as in original)
    state.append(remaining_moves / 2)

    # smell cooldown (normalized by 5)
    max_distance = np.sqrt(field_width**2 + field_height**2)
    state.append(prey.smell_cd / float(SMELL_COOLDOWN_TURNS))

    # per-hunter features (if detected): distance (norm), direction (x,y), aware flag; else zeros
    for hunter in hunters:
        if hunter.detected:
            hx, hy = hunter.position
            distance = np.linalg.norm(prey.position - hunter.position)
            normalized_distance = distance / max_distance
            state.append(normalized_distance)
            if distance > 0:
                direction_x = (hx - prey_x) / distance
                direction_y = (hy - prey_y) / distance
            else:
                direction_x = 0.0
                direction_y = 0.0
            state.extend([direction_x, direction_y])
            state.append(1.0 if hunter.aware else 0.0)
        else:
            state.extend([0.0, 0.0, 0.0, 0.0])

    # distances to walls (normalized)
    dist_to_left = prey_x / field_width
    dist_to_right = (field_width - 1 - prey_x) / field_width
    dist_to_top = prey_y / field_height
    dist_to_bottom = (field_height - 1 - prey_y) / field_height
    state.extend([dist_to_left, dist_to_right, dist_to_top, dist_to_bottom])

    return np.array(state, dtype=np.float32)

def calculate_reward(prey: Prey, hunters: List[Hunter], caught: bool, survived_turn: int, total_turns: int):
    # Simple shaping: big penalty if caught, big reward if fully survived, small step reward otherwise
    if caught:
        return -300.0
    if survived_turn == total_turns:
        return 100.0
    return 1.0


# =========================
# Training Loop
# =========================

def train_dqn_agent(episodes=2000, max_turns=MAX_TURNS_DEFAULT, batch_size=32, target_sync_every=50, seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    action_names = ["up", "down", "left", "right", "still", "smell"]
    action_size = len(action_names)
    state_size = 27  # IMPORTANT: must stay consistent with representation above

    agent = DQNAgent(state_size, action_size)
    scores = []
    survival_flags = []
    last_turns = []

    for episode in range(episodes):
        player, hunters = spawning(field_height=FIELD_HEIGHT, field_width=FIELD_WIDTH)
        flared = False
        state = get_state_representation(player, hunters, FIELD_WIDTH, FIELD_HEIGHT, remaining_moves=2)

        total_reward = 0.0
        caught = False
        done = False

        for turn in range(max_turns):
            # Prey may have multiple moves per turn if movement_speed>1
            for i in range(player.movement_speed):
                action_idx = agent.act(state, player, FIELD_WIDTH, FIELD_HEIGHT)
                action = action_names[action_idx]

                # Execute prey action
                player.move(action, hunters)

                # Prey sees own flares nearby -> mark as seen (for state)
                if player.flares:
                    for flare in player.flares:
                        if np.linalg.norm(player.position - flare) <= 1 and not any(np.array_equal(flare, seen) for seen in player.seen_flares):
                            player.seen_flares.append(flare)

                # Check catch right after prey move
                caught = any(np.array_equal(h.position, player.position) for h in hunters)
                done = caught or (turn + 1 == max_turns)

                next_state = get_state_representation(player, hunters, FIELD_WIDTH, FIELD_HEIGHT, remaining_moves=(player.movement_speed - (i + 1)))
                reward = calculate_reward(player, hunters, caught, survived_turn=turn + 1, total_turns=max_turns)
                total_reward += reward

                agent.remember(state, action_idx, reward, next_state, done)
                state = next_state
                if done:
                    last_turns.append(turn + 1)
                    break
            if done:
                break

            # Handle flares: if prey stayed near its seen flares -> hunters become aware
            if player.flares or player.seen_flares:
                for flare in player.seen_flares:
                    if np.linalg.norm(player.position - flare) <= 1:
                        flared = True
                # clear flares each turn (simple mechanic)
                player.flares.clear()
                player.seen_flares.clear()

            # Hunters move
            for hunter in hunters:
                if flared:
                    hunter.aware = True
                hunter.move(player.get_position(), hunters, player)
            flared = False

            # After hunters move, update detection flags
            update_detection(player, hunters)

            # Cooldowns
            player.decay_cooldowns()

            # Check catch
            caught = any(np.array_equal(h.position, player.position) for h in hunters)
            done = caught or (turn + 1 == max_turns)

            # Build next state and (optionally) an extra reward step
            next_state = get_state_representation(player, hunters, FIELD_WIDTH, FIELD_HEIGHT, remaining_moves=0)
            # You could add extra shaping here if desired

            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state

            # Learn periodically
            if len(agent.memory) >= batch_size and (turn % 5 == 0):
                agent.replay(batch_size)

        # Sync target network periodically
        if (episode + 1) % target_sync_every == 0:
            agent.update_target_network()

        scores.append(total_reward)
        survival_flags.append(0 if caught else 1)

        if (episode + 1) % 100 == 0:
            avg_score = float(np.mean(scores[-100:]))
            avg_turns = float(np.mean(last_turns[-100:])) if last_turns else 0.0
            survival_rate = float(np.mean(survival_flags[-100:]) * 100.0)
            print(f"[Ep {episode+1}] AvgScore: {avg_score:.2f} | Survival: {survival_rate:.1f}% | Eps: {agent.epsilon:.3f} | AvgTurns: {avg_turns:.1f}")

    # ===== Save weights & meta =====
    torch.save(agent.q_network.state_dict(), "prey_dqn_model.pth")
    with open("prey_dqn_meta.json", "w") as f:
        json.dump({"state_size": state_size, "action_size": action_size}, f)
    print("Saved model -> prey_dqn_model.pth and meta -> prey_dqn_meta.json")

    return agent, scores, survival_flags


# =========================
# Evaluation
# =========================

def evaluate(agent: DQNAgent, episodes=50, max_turns=MAX_TURNS_DEFAULT, render: bool = False):
    action_names = ["up", "down", "left", "right", "still", "smell"]
    survival = []
    total_turns = []

    # greedy eval
    eps_backup = agent.epsilon
    agent.epsilon = 0.0

    for ep in range(episodes):
        player, hunters = spawning(field_height=FIELD_HEIGHT, field_width=FIELD_WIDTH)
        update_detection(player, hunters)
        state = get_state_representation(player, hunters, FIELD_WIDTH, FIELD_HEIGHT, remaining_moves=2)
        caught = False

        for turn in range(max_turns):
            a_idx = agent.act(state, player, FIELD_WIDTH, FIELD_HEIGHT)
            a = action_names[a_idx]
            player.move(a, hunters)

            if player.flares:
                for flare in player.flares:
                    if np.linalg.norm(player.position - flare) <= 1 and not any(np.array_equal(flare, seen) for seen in player.seen_flares):
                        player.seen_flares.append(flare)

            # Hunters chase & detect
            for h in hunters:
                h.move(player.get_position(), hunters, player)
            update_detection(player, hunters)
            player.decay_cooldowns()

            if any(np.array_equal(h.position, player.position) for h in hunters):
                caught = True
                break

            state = get_state_representation(player, hunters, FIELD_WIDTH, FIELD_HEIGHT, remaining_moves=0)

        survival.append(0 if caught else 1)
        total_turns.append(turn + 1)

    agent.epsilon = eps_backup
    print(f"Eval — Survival rate: {np.mean(survival)*100:.1f}% | Avg turns: {np.mean(total_turns):.1f}")
    return float(np.mean(survival)), float(np.mean(total_turns))


# =========================
# CLI
# =========================

if __name__ == "__main__":
    # Example quick train (reduce episodes for a smoke test)
    trained_agent, scores, flags = train_dqn_agent(episodes=200, max_turns=MAX_TURNS_DEFAULT)
    evaluate(trained_agent, episodes=20, max_turns=MAX_TURNS_DEFAULT)


