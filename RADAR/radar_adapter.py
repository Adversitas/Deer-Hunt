
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ["UP","DOWN","LEFT","RIGHT","STAY","SMELL"]

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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_size, 256, action_size).to(self.device)
        self.q_network.eval()

    def load_weights(self, path):
        state = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(state)
        self.q_network.eval()

    def act_greedy(self, state_vec, valid_actions=None):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        if valid_actions is None:
            return int(np.argmax(q_values))
        mask = np.full_like(q_values, -np.inf, dtype=float)
        mask[valid_actions] = q_values[valid_actions]
        return int(np.argmax(mask))

class PreyShim:
    def __init__(self, position, smell_cd=0, flares=None, seen_flares=None, movement_speed=1):
        self.position = np.array(position, dtype=int)
        self.smell_cd = smell_cd
        self.flares = [np.array(f) for f in (flares or [])]
        self.seen_flares = [np.array(f) for f in (seen_flares or [])]
        self.movement_speed = movement_speed

class HunterShim:
    def __init__(self, position, detected=False, aware=False):
        self.position = np.array(position, dtype=int)
        self.detected = detected
        self.aware = aware

def get_state_representation(prey, hunters, field_width, field_height, remaining_moves):
    state = []
    prey_x, prey_y = prey.position
    state.extend([prey_x / field_width, prey_y / field_height])
    flare_count = 0
    for flare in prey.flares:
        if any(np.array_equal(flare, seen) for seen in prey.seen_flares):
            state.extend([flare[0] / field_width, flare[1] / field_height])
            flare_count += 1
        else:
            state.extend([0, 0])
            flare_count += 1
    while flare_count != 3:
        state.extend([0, 0])
        flare_count += 1
    visible_hunters = 0
    for hunter in hunters:
        if getattr(hunter, "detected", False):
            visible_hunters += 1
    visible_count = visible_hunters / max(1, len(hunters))
    state.append(visible_count)
    state.append(remaining_moves / 2)
    max_distance = np.sqrt(field_width**2 + field_height**2)
    state.append(prey.smell_cd / 5)
    for hunter in hunters:
        if getattr(hunter, "detected", False):
            hunter_x, hunter_y = hunter.position
            distance = np.linalg.norm(prey.position - hunter.position)
            normalized_distance = distance / max_distance
            state.append(normalized_distance)
            if distance > 0:
                direction_x = (hunter_x - prey_x) / distance
                direction_y = (hunter_y - prey_y) / distance
            else:
                direction_x, direction_y = 0, 0
            state.extend([direction_x, direction_y])
            state.append(1.0 if getattr(hunter, "aware", False) else 0.0)
        else:
            state.extend([0.0, 0.0, 0.0, 0.0])
    dist_to_left = prey_x / field_width
    dist_to_right = (field_width - 1 - prey_x) / field_width
    dist_to_top = prey_y / field_height
    dist_to_bottom = (field_height - 1 - prey_y) / field_height
    state.extend([dist_to_left, dist_to_right, dist_to_top, dist_to_bottom])
    return np.array(state, dtype=np.float32)

def compute_detection_flags(prey_pos, hunters_pos, sight_radius=3):
    pr = np.array(prey_pos)
    flags = []
    for h in hunters_pos:
        d = np.linalg.norm(pr - np.array(h))
        detected = d <= sight_radius
        aware = detected  # simple proxy; tune if your training used something else
        flags.append((detected, aware))
    return flags

def valid_actions_from_bounds(prey_pos, rows, cols, smell_cd=0):
    x, y = prey_pos
    valid = []
    if y > 0: valid.append(0)               # up
    if y < rows - 1: valid.append(1)        # down
    if x > 0: valid.append(2)               # left
    if x < cols - 1: valid.append(3)        # right
    valid.append(4)                          # stay
    if smell_cd == 0: valid.append(5)       # smell
    return valid

def load_agent(weights_path, input_size):
    agent = DQNAgent(state_size=input_size, action_size=len(ACTIONS))
    agent.load_weights(weights_path)
    return agent

def policy_action(agent, gui_state, remaining_moves=2, sight_radius=3):
    rows, cols = gui_state["grid_size"]
    prey = PreyShim(gui_state["animal"], smell_cd=0, flares=[], seen_flares=[], movement_speed=1)
    hunters = []
    flags = compute_detection_flags(gui_state["animal"], gui_state["hunters"], sight_radius=sight_radius)
    for (pos, (det, aw)) in zip(gui_state["hunters"], flags):
        hunters.append(HunterShim(pos, detected=det, aware=aw))
    state_vec = get_state_representation(prey, hunters, cols, rows, remaining_moves)
    valid = valid_actions_from_bounds(gui_state["animal"], rows, cols, smell_cd=0)
    idx = agent.act_greedy(state_vec, valid_actions=valid)
    return ACTIONS[idx]
