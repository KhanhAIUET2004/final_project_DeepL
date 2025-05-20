import numpy as np
from policy_net import PolicyGradientCNN, ActorCNN
import torch
from torch.distributions import Categorical
import random

class Agents:

    def __init__(self):
        """
            TODO:
        """
        self.state = None
        self.packages = {}

        self.model_pth = "model_weigth/map1/final_map1.pth"

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_model = ActorCNN()
        if self.model_pth is not None:
            try:
                
                checkpoint = torch.load(self.model_pth, map_location=self.device)
                self.policy_model.load_state_dict(checkpoint['actor_state_dict'])
                self.policy_model.eval()
                self.policy_model.to(self.device)
                print(f'Load model from {self.model_pth} successfully')
            except Exception as e:
                print(f'Error loading model: {e}')
                print('Model path is invalid or model structure mismatch')
        else: 
            print('Model path is None')


    def init_agents(self, state):
        """
            TODO:
        """
        self.state = state

    def get_actions(self, state):
        self.state = self.state_to_agent(state)

        actions = []

        with torch.no_grad(): # No gradient needed for inference
            for i in range(len(state['robots'])):
                # Build state tensor specific to robot i
                input = self.convert_state_to_actor_input(self.state, self.packages, i)
                robot_state_tensor = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
                robot_state_tensor = robot_state_tensor.to(self.device)

                # Get action probability distribution from the model
                probs = self.policy_model(robot_state_tensor)
                probs = torch.softmax(probs, dim=-1)
                
                action_dist = Categorical(probs=probs)
                action_index_tensor = action_dist.sample()

                # Decode the action index into move and package actions
                action_index = action_index_tensor.item()
                action_tuple = ActorCNN.decode_action(action_index)
                actions.append(action_tuple)


        return actions
    def state_to_agent(self, state):
        if state['time_step'] == 0:
            self.packages = {}
        packages_in_carry = set()
        if len(state['packages']) > 0:
            for package in state['packages']:
                package_id = package[0]
                x, y, target_x, target_y, start_time, deadline = package[1:]
                status = 'waiting'
                self.packages[package_id] = {
                    'x': x,
                    'y': y,
                    'target_x': target_x,
                    'target_y': target_y,
                    'start_time': start_time,
                    'deadline': deadline,
                    'status': status
                }
        for robot in state['robots']:
            package_carry = robot[2]
            if package_carry != 0:
                self.packages[package_carry]['status'] = 'carrying'
                packages_in_carry.add(package_carry)
        for package_id in self.packages:
            if self.packages[package_id]['status'] == 'carrying' and package_id not in packages_in_carry:
                self.packages[package_id]['status'] = 'Delivered'
        state = {
            'robots': state['robots'],
            'packages':[(package_id,
                            self.packages[package_id]['x'],
                            self.packages[package_id]['y'],
                                self.packages[package_id]['target_x'],
                                self.packages[package_id]['target_y'],
                                    self.packages[package_id]['start_time'],
                                    self.packages[package_id]['deadline'],
                                        self.packages[package_id]['status'])for package_id in self.packages],
            'map': state['map'],
            'time_step': state['time_step'],
        }
        return state
    def convert_state_to_actor_input(self, global_state_dict, all_packages_data, current_agent_idx):
        """
        Chuyển đổi global state và thông tin package thành local observation cho một Actor cụ thể.

        Args:
            global_state_dict (dict): Output từ env.get_state().
            all_packages_data (dict): Dictionary chứa thông tin đầy đủ của tất cả các package.
            current_agent_idx (int): Index (0-based) của agent hiện tại trong danh sách robots.
            map_dims (tuple): (MAP_HEIGHT, MAP_WIDTH).

        Returns:
            np.ndarray: Local observation tensor cho Actor.
        """
        MAP_HEIGHT, MAP_WIDTH = (len(self.state['map'][0]), len(self.state['map'][0]))
        num_actor_channels = 6 # Theo đề xuất 6 kênh ở trên
        actor_obs = np.zeros((num_actor_channels, MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
        current_time = global_state_dict['time_step']

        # --- Channel 0: Map Obstacles ---
        grid_map = np.array(global_state_dict['map'])
        actor_obs[0, :, :] = grid_map[:MAP_HEIGHT, :MAP_WIDTH]

        # --- Lấy thông tin robots ---
        robots_info = global_state_dict['robots']  # list of (x, y, carrying_package_id)
        
        # Dữ liệu của agent hiện tại
        agent_data = robots_info[current_agent_idx]
        agent_x_1based, agent_y_1based, agent_carrying_pkg_id = agent_data
        agent_r, agent_c = agent_x_1based - 1, agent_y_1based - 1

        # --- Channel 1: Vị trí của Agent hiện tại ---
        if 0 <= agent_r < MAP_HEIGHT and 0 <= agent_c < MAP_WIDTH:
            actor_obs[1, agent_r, agent_c] = 1.0

        # --- Channel 2: Vị trí của các Agent khác ---
        for i, other_robot_data in enumerate(robots_info):
            if i == current_agent_idx:
                continue # Bỏ qua chính agent này
            
            other_x_1based, other_y_1based, _ = other_robot_data
            other_r, other_c = other_x_1based - 1, other_y_1based - 1
            if 0 <= other_r < MAP_HEIGHT and 0 <= other_c < MAP_WIDTH:
                actor_obs[2, other_r, other_c] = 1.0
                
        # --- Xử lý Packages ---

        # Chuẩn hóa urgency (có thể dùng hàm riêng nếu logic phức tạp)
        def get_normalized_urgency(pkg_deadline, current_time_step):
            time_to_deadline = max(0, pkg_deadline - current_time_step)
            urgency_val = 1.0 / (time_to_deadline + 1e-5 + 0.1) # Giống critic
            # Ví dụ chuẩn hóa: urgency_val / (urgency_val + K), K=4
            normalized = urgency_val / (urgency_val + 4.0) 
            return min(1.0, max(0.0, normalized)) # Đảm bảo trong [0,1]

        # Xử lý package mà agent hiện tại đang mang
        if agent_carrying_pkg_id != 0:
            if agent_carrying_pkg_id in all_packages_data:
                carried_pkg_details = all_packages_data[agent_carrying_pkg_id]
                
                # --- Channel 4: Vị trí đích của Package mà Agent hiện tại đang mang ---
                pt_x_1based = carried_pkg_details['target_x']
                pt_y_1based = carried_pkg_details['target_y']
                pt_r, pt_c = pt_x_1based - 1, pt_y_1based - 1
                if 0 <= pt_r < MAP_HEIGHT and 0 <= pt_c < MAP_WIDTH:
                    actor_obs[4, pt_r, pt_c] = 1.0
                
                # --- Channel 5: Mức độ khẩn cấp (Cách 2 - Agent-specific) ---
                urgency = get_normalized_urgency(carried_pkg_details['deadline'], current_time)
                if 0 <= agent_r < MAP_HEIGHT and 0 <= agent_c < MAP_WIDTH:
                    actor_obs[5, agent_r, agent_c] = max(actor_obs[5, agent_r, agent_c], urgency)


        # Xử lý các package khác (chủ yếu là 'waiting' cho agent không mang hàng)
        for pkg_id, pkg_details in all_packages_data.items():
            status = pkg_details['status']
            
            if status == 'waiting':
                # --- Channel 3: Vị trí các Package đang chờ (Waiting) ---
                p_x_1based, p_y_1based = pkg_details['x'], pkg_details['y']
                p_r, p_c = p_x_1based - 1, p_y_1based - 1
                if 0 <= p_r < MAP_HEIGHT and 0 <= p_c < MAP_WIDTH:
                    actor_obs[3, p_r, p_c] = 1.0
                
                # --- Channel 5: Mức độ khẩn cấp (Cách 2 - Agent-specific) ---
                # Nếu agent không mang hàng, đặt urgency của package 'waiting' tại vị trí HIỆN TẠI của package đó
                # (để agent biết package nào gần/khẩn cấp để nhặt)
                if agent_carrying_pkg_id == 0: # Chỉ xem xét nếu agent đang rảnh
                    urgency = get_normalized_urgency(pkg_details['deadline'], current_time)
                    if 0 <= p_r < MAP_HEIGHT and 0 <= p_c < MAP_WIDTH:
                        actor_obs[5, p_r, p_c] = max(actor_obs[5, p_r, p_c], urgency)
        return actor_obs

class RandomAgent:
    def __init__(self, num_actions=15):
        self.num_actions = num_actions

    def init_agents(self, state):
        pass

    def get_actions(self, state):
        num_robots = len(state['robots'])
        actions = []
        for _ in range(num_robots):
            action_index = random.randint(0, self.num_actions - 1)
            action_tuple = ActorCNN.decode_action(action_index)
            actions.append(action_tuple)
        return actions
