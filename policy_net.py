import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple, Optional
import numpy as np


class PolicyGradientCNN(nn.Module):
    """
    Mạng Policy Gradient sử dụng CNN xử lý đầu vào tensor (batch, 1, 7, 20, 20)
    và đưa ra xác suất cho 15 hành động kết hợp.
    Bao gồm hàm decode_action bên trong.
    """
    # Định nghĩa các hằng số hành động dưới dạng thuộc tính lớp
    MOVE_ACTIONS = ['S', 'L', 'R', 'U', 'D']
    PACKAGE_ACTIONS = ['0', '1', '2']
    NUM_MOVE_ACTIONS = len(MOVE_ACTIONS)
    NUM_PACKAGE_ACTIONS = len(PACKAGE_ACTIONS)
    NUM_COMBINED_ACTIONS = NUM_MOVE_ACTIONS * NUM_PACKAGE_ACTIONS # 5 * 3 = 15

    def __init__(self, input_shape=(1, 6, 20, 20), hidden_dim=256):
        """
        Khởi tạo mạng với các lớp CNN và FC.
        Args:
            input_shape (tuple): Kích thước của một state (outer_dim, channel, height, width).
                                 Ví dụ: (1, 6, 20, 20).
                                 Lưu ý: CNN sẽ xử lý input_shape[1:] (channel, height, width).
            hidden_dim (int): Số chiều của lớp ẩn fully connected trước output.
        """
        super(PolicyGradientCNN, self).__init__()

        self._input_shape = input_shape
        self.cnn_input_channels = input_shape[1] # 6
        self.cnn_input_height = input_shape[2]   # 20
        self.cnn_input_width = input_shape[3]    # 20

        self.hidden_dim = hidden_dim
        self.action_dim = self.NUM_COMBINED_ACTIONS # Sử dụng thuộc tính lớp

        # Định nghĩa các lớp CNN
        self.conv1 = nn.Conv2d(in_channels=self.cnn_input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Tính toán kích thước đầu vào cho lớp Fully Connected đầu tiên sau khi làm phẳng
        dummy_input = torch.zeros(1, self.cnn_input_channels, self.cnn_input_height, self.cnn_input_width)
        dummy_output = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(dummy_input))))))
        self.cnn_output_flattened_size = torch.flatten(dummy_output, 1).size(1)
        print(f"Kích thước sau CNN và làm phẳng: {self.cnn_output_flattened_size}") # Nên là 64 * 5 * 5 = 1600


        # Các lớp fully connected (linear) sau CNN
        self.fc1 = nn.Linear(self.cnn_output_flattened_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.action_dim)

    @staticmethod
    def decode_action(action_index: int) -> tuple:
        """
        Chuyển đổi chỉ số hành động (0-14) thành tuple (move_action, package_action).
        Phương thức tĩnh (static method) của lớp PolicyGradientCNN.
        """
        if not 0 <= action_index < PolicyGradientCNN.NUM_COMBINED_ACTIONS: # Truy cập hằng số qua tên lớp
            raise ValueError(f"Chỉ số hành động không hợp lệ: {action_index}. Phải nằm trong khoảng 0 đến {PolicyGradientCNN.NUM_COMBINED_ACTIONS - 1}.")

        move_index = action_index // PolicyGradientCNN.NUM_PACKAGE_ACTIONS # Truy cập hằng số qua tên lớp
        package_index = action_index % PolicyGradientCNN.NUM_PACKAGE_ACTIONS # Truy cập hằng số qua tên lớp

        move_action = PolicyGradientCNN.MOVE_ACTIONS[move_index] # Truy cập hằng số qua tên lớp
        package_action = PolicyGradientCNN.PACKAGE_ACTIONS[package_index] # Truy cập hằng số qua tên lớp

        return (move_action, package_action)


    def forward(self, x):
        """
        Pass dữ liệu qua mạng CNN và FC.
        Args:
            x (torch.Tensor): Tensor đầu vào có shape (batch_size, 1, 7, 20, 20).
        Returns:
            torch.Tensor: Tensor xác suất cho mỗi hành động, shape (batch_size, 15).
        """
        # Loại bỏ chiều đơn ở vị trí 1
        x = x.squeeze(1) # Shape mới: (batch_size, 7, 20, 20)

        # Pass qua các lớp CNN
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Làm phẳng
        x = torch.flatten(x, 1)

        # Pass qua các lớp FC
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Áp dụng softmax
        return F.softmax(x, dim=-1)

class ActorCNN(nn.Module):
    def __init__(self, input_channels=6, num_actions=15):
        """
        Actor Network sử dụng CNN để xử lý local observation và đưa ra logit hành động.

        Args:
            input_channels (int): Số lượng kênh của local observation (ví dụ: 6).
            num_actions (int): Tổng số hành động mà agent có thể thực hiện (ví dụ: 5 di chuyển * 3 package_ops = 15).
        """
        super(ActorCNN, self).__init__()
        self.input_channels = input_channels
        self.num_actions = num_actions

        # Convolutional layers (Kiến trúc ví dụ, có thể điều chỉnh)
        # Input: (batch_size, input_channels, MAP_HEIGHT, MAP_WIDTH)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling

        # Fully connected layers
        # Kích thước sau pooling là (batch_size, 128, 1, 1) -> flatten thành (batch_size, 128)
        self.fc1 = nn.Linear(128, 128) # Giữ nguyên 128 hoặc giảm xuống
        self.fc_actor_head = nn.Linear(128, num_actions) # Output logits cho các hành động

    def forward(self, local_observation):
        """
        Forward pass của Actor network.

        Args:
            local_observation (torch.Tensor): Local observation của agent,
                                             shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Logits cho các hành động, shape (batch_size, num_actions).
        """
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(local_observation)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x) # (batch_size, 128, 1, 1)

        # Flatten
        x = x.view(x.size(0), -1) # (batch_size, 128)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        action_logits = self.fc_actor_head(x) # (batch_size, num_actions)

        return action_logits

    @staticmethod
    def decode_action(action_index):
        """
        Chuyển đổi một action index (0-14) thành (move_action_str, package_action_str).
        Ví dụ: 0 -> ('S', '0')
        """
        if not (0 <= action_index < 15):
            raise ValueError("Action index phải nằm trong khoảng [0, 14]")

        move_actions = ['S', 'L', 'R', 'U', 'D']  # Stay, Left, Right, Up, Down
        package_actions = ['0', '1', '2']          # Do_nothing, Pickup, Drop

        move_idx = action_index // len(package_actions) # 0, 1, 2, 3, 4
        package_idx = action_index % len(package_actions) # 0, 1, 2

        return move_actions[move_idx], package_actions[package_idx]

    @staticmethod
    def encode_action(move_action_str, package_action_str):
        """
        Chuyển đổi (move_action_str, package_action_str) thành action index (0-14).
        Ví dụ: ('S', '0') -> 0
        """
        move_actions = ['S', 'L', 'R', 'U', 'D']
        package_actions = ['0', '1', '2']

        try:
            move_idx = move_actions.index(move_action_str)
            package_idx = package_actions.index(package_action_str)
        except ValueError:
            raise ValueError(f"Hành động không hợp lệ: ({move_action_str}, {package_action_str})")

        return move_idx * len(package_actions) + package_idx