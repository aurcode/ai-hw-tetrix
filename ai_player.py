# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pygame",
#     "torch",
#     "torchvision",
# ]
# ///
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pygame # For key constants
import os # For path joining

# --- Constants and Configuration ---
pygame.init() # Initialize Pygame to use its key constants

BOARD_WIDTH = 10
BOARD_HEIGHT = 20
MODEL_FILENAME = "tetris_action_predictor.pth"

# Define standard Tetris shapes
ALL_SHAPES = ['IBlock', 'OBlock', 'TBlock', 'SBlock', 'ZBlock', 'LBlock', 'JBlock']
SHAPE_TO_IDX = {shape: i for i, shape in enumerate(ALL_SHAPES)}

# Define the actions our NN will predict
POSSIBLE_ACTIONS = ["Left", "Right", "Rotate", "SoftDrop", "HardDrop"]
ACTION_TO_LABEL = {action: i for i, action in enumerate(POSSIBLE_ACTIONS)}
LABEL_TO_ACTION = {i: action for action, i in ACTION_TO_LABEL.items()}
NUM_ACTIONS = len(POSSIBLE_ACTIONS)

# Map Pygame keys (from data) to our defined action names
PYGAME_ACTION_MAP = {
    pygame.K_LEFT: "Left",
    pygame.K_RIGHT: "Right",
    pygame.K_UP: "Rotate",
    pygame.K_DOWN: "SoftDrop",
    pygame.K_SPACE: "HardDrop"
}

# Reverse mapping: Action names to Pygame keys (for NNAIPlayer output)
ACTION_NAME_TO_PYGAME_KEY = {v: k for k, v in PYGAME_ACTION_MAP.items()}



# --- Feature Engineering Functions (used by training and NNAIPlayer) ---
def get_board_features(board_state_np):
    """Calculates features from the board state."""
    if board_state_np.shape != (BOARD_HEIGHT, BOARD_WIDTH):
        raise ValueError(f"Expected board shape ({BOARD_HEIGHT},{BOARD_WIDTH}), got {board_state_np.shape}")

    column_heights = np.zeros(BOARD_WIDTH, dtype=float)
    for col in range(BOARD_WIDTH):
        for row in range(BOARD_HEIGHT):
            if board_state_np[row, col] == 1:
                column_heights[col] = BOARD_HEIGHT - row
                break
    aggregate_height = np.sum(column_heights)
    num_holes = 0
    for col in range(BOARD_WIDTH):
        col_has_block = False
        for row in range(BOARD_HEIGHT):
            if board_state_np[row, col] == 1:
                col_has_block = True
            elif col_has_block and board_state_np[row, col] == 0:
                num_holes += 1
    bumpiness = 0
    for i in range(BOARD_WIDTH - 1):
        bumpiness += abs(column_heights[i] - column_heights[i+1])
    completed_lines = 0
    for row in range(BOARD_HEIGHT):
        if np.all(board_state_np[row, :] == 1):
            completed_lines += 1
            
    norm_column_heights = column_heights / BOARD_HEIGHT
    norm_aggregate_height = aggregate_height / (BOARD_WIDTH * BOARD_HEIGHT)
    norm_num_holes = num_holes / (BOARD_WIDTH * BOARD_HEIGHT)
    norm_bumpiness = bumpiness / (BOARD_WIDTH * BOARD_HEIGHT)
    norm_completed_lines = completed_lines / BOARD_HEIGHT
    
    return (norm_column_heights, norm_aggregate_height, norm_num_holes, 
            norm_bumpiness, norm_completed_lines)

def featurize_training_data(data_point):
    """Converts a single raw data point (from training data list) into a flat feature vector and a label."""
    board_state_list = data_point['board_state']
    board_state_np = np.array(board_state_list, dtype=float)

    (norm_col_heights, norm_agg_h, norm_holes, 
     norm_bump, norm_lines) = get_board_features(board_state_np)

    current_shape_one_hot = np.zeros(len(ALL_SHAPES), dtype=float)
    # Handle 'LineBlock' and 'SquareBlock' for training data if it exists
    current_shape_name = data_point['current_block_shape']
    if current_shape_name == 'LineBlock': current_shape_name = 'IBlock'
    if current_shape_name == 'SquareBlock': current_shape_name = 'OBlock'
    shape_idx = SHAPE_TO_IDX.get(current_shape_name)
    if shape_idx is not None:
        current_shape_one_hot[shape_idx] = 1.0

    rotation_norm = float(data_point['current_block_rotation']) / 360.0
    pos_x_norm = float(data_point['current_block_pos'][0]) / BOARD_WIDTH
    pos_y_norm = float(data_point['current_block_pos'][1]) / BOARD_HEIGHT
    
    next_shape_one_hot = np.zeros(len(ALL_SHAPES), dtype=float)
    next_shape_name = data_point['next_block_shape']
    if next_shape_name == 'LineBlock': next_shape_name = 'IBlock'
    if next_shape_name == 'SquareBlock': next_shape_name = 'OBlock'
    next_shape_idx = SHAPE_TO_IDX.get(next_shape_name)
    if next_shape_idx is not None:
        next_shape_one_hot[next_shape_idx] = 1.0
        
    features = np.concatenate([
        norm_col_heights,
        np.array([norm_agg_h, norm_holes, norm_bump, norm_lines]),
        current_shape_one_hot,
        np.array([rotation_norm, pos_x_norm, pos_y_norm]),
        next_shape_one_hot
    ]).astype(np.float32)

    action_key = data_point['action']
    action_name = PYGAME_ACTION_MAP.get(action_key)
    label = -1 
    if action_name and action_name in ACTION_TO_LABEL:
        label = ACTION_TO_LABEL[action_name]
    
    return features, label

# --- PyTorch Dataset ---
class TetrisDataset(Dataset):
    def __init__(self, data_points):
        self.features_list = []
        self.labels_list = []
        
        for dp in data_points:
            features, label = featurize_training_data(dp) # Use the renamed function
            if label != -1: 
                self.features_list.append(features)
                self.labels_list.append(label)
            else:
                print(f"Warning: Skipping training data point due to unmapped action: {dp.get('action')}")
        
        if not self.features_list:
            raise ValueError("No valid data points found for training after featurization and action mapping.")

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        return torch.tensor(self.features_list[idx], dtype=torch.float), \
               torch.tensor(self.labels_list[idx], dtype=torch.long)

# --- PyTorch Model ---
# Ensure raw_data_points is not empty before accessing its first element

import json
file_path = "tetris_training_data.json"
with open(file_path, 'r') as f:
    raw_data_points = json.load(f)

# Keep only the first 2523 data points to match the original script's behavior
if isinstance(raw_data_points, list):
    raw_data_points = raw_data_points[:2523]  # Limit to first 2523 points

if not raw_data_points:
    raise ValueError("raw_data_points is empty. Cannot determine INPUT_FEATURES_SIZE.")
sample_features_for_size_calc, _ = featurize_training_data(raw_data_points[0])
INPUT_FEATURES_SIZE = len(sample_features_for_size_calc)

class TetrisActionPredictor(nn.Module):
    def __init__(self, input_size, num_actions):
        super(TetrisActionPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),  # Increased width
            nn.ReLU(),
            nn.Dropout(0.3),             # Adjusted dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),         # Extra layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)

# --- Training and Evaluation ---
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    print(f"\n--- Starting Training (Input features: {INPUT_FEATURES_SIZE}, Output actions: {NUM_ACTIONS}) ---")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
        avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print("--- Training Finished ---")

def evaluate_model(model, dataloader, criterion):
    print("\n--- Starting Evaluation ---")
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    if dataloader is None or len(dataloader) == 0 : # Check if dataloader is None or empty
        print("Evaluation dataloader is empty or None. Skipping evaluation.")
        return 0.0
        
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
    print(f"Evaluation Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print("--- Evaluation Finished ---")
    return accuracy

# --- Placeholder Block Class (for NNAIPlayer integration) ---
# This class should ideally match the structure of the Block class in your actual game (block.py)
class Block: # Renamed to avoid conflict if block.py is imported directly
    def __init__(self, shape_name, rotation_degrees, x_pos, y_pos, points=None):
        self.shape_name = shape_name # Use shape_name to match user's block.py
        self.rotation = rotation_degrees 
        self.x = x_pos 
        self.y = y_pos 
        self.points = points if points is not None else [] 

# --- NNAIPlayer Class ---
class NNAIPlayer:
    def __init__(self, grid_width, grid_height, model_path=MODEL_FILENAME):
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        self.board_width_const = BOARD_WIDTH
        self.board_height_const = BOARD_HEIGHT
        self.all_shapes_const = ALL_SHAPES
        self.shape_to_idx_const = SHAPE_TO_IDX
        self.label_to_action_const = LABEL_TO_ACTION
        self.action_name_to_pygame_key_const = ACTION_NAME_TO_PYGAME_KEY

        self.input_features_size = INPUT_FEATURES_SIZE 
        self.num_actions = NUM_ACTIONS 

        self.model = TetrisActionPredictor(self.input_features_size, self.num_actions)
        
        if os.path.exists(model_path):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.to(device) 
                self.model.eval() 
                print(f"NNAIPlayer initialized. Loaded model from {model_path} to {device}.")
            except Exception as e:
                print(f"Error loading model: {e}. NNAIPlayer will use an untrained model.")
        else:
            print(f"Warning: Model file {model_path} not found. NNAIPlayer will use an untrained model.")
        
        print(f"NNAIPlayer initialized for a {grid_width}x{grid_height} grid.")


    def _featurize_live_game_state(self, game_board_state_np, current_block_obj, next_block_obj):
        """Converts live game state (board, current block obj, next block obj) into a feature vector."""
        
        (norm_col_heights, norm_agg_h, norm_holes, 
         norm_bump, norm_lines) = get_board_features(game_board_state_np)

        current_shape_one_hot = np.zeros(len(self.all_shapes_const), dtype=float)
        current_shape_name = current_block_obj.shape_name # MODIFIED: Use .shape_name
        if current_shape_name == 'LineBlock': current_shape_name = 'IBlock' 
        if current_shape_name == 'SquareBlock': current_shape_name = 'OBlock' # ADDED: Mapping for SquareBlock
        shape_idx = self.shape_to_idx_const.get(current_shape_name)
        if shape_idx is not None:
            current_shape_one_hot[shape_idx] = 1.0

        rotation_norm = float(current_block_obj.rotation) / 360.0
        pos_x_norm = float(current_block_obj.x) / self.board_width_const
        pos_y_norm = float(current_block_obj.y) / self.board_height_const
        
        next_shape_one_hot = np.zeros(len(self.all_shapes_const), dtype=float)
        next_shape_name = next_block_obj.shape_name # MODIFIED: Use .shape_name
        if next_shape_name == 'LineBlock': next_shape_name = 'IBlock' 
        if next_shape_name == 'SquareBlock': next_shape_name = 'OBlock' # ADDED: Mapping for SquareBlock
        next_shape_idx = self.shape_to_idx_const.get(next_shape_name)
        if next_shape_idx is not None:
            next_shape_one_hot[next_shape_idx] = 1.0
            
        features = np.concatenate([
            norm_col_heights,
            np.array([norm_agg_h, norm_holes, norm_bump, norm_lines]),
            current_shape_one_hot,
            np.array([rotation_norm, pos_x_norm, pos_y_norm]),
            next_shape_one_hot
        ]).astype(np.float32)
        
        return features

    def get_next_move(self, game_board_state, current_block, next_block):
        """
        Determines the next move for the AI using the loaded neural network.
        Args:
            game_board_state (np.array): Current state of the game board.
            current_block (Block): The current falling block object (should have .shape_name, .rotation, .x, .y).
            next_block (Block): The next block object (should have .shape_name).
        Returns:
            list: A list containing a single Pygame key for the predicted action, 
                  or an empty list if no action is predicted or an error occurs.
        """
        try:
            # Ensure current_block and next_block have the 'shape_name' attribute
            if not hasattr(current_block, 'shape_name') or not hasattr(next_block, 'shape_name'):
                print("Error: current_block or next_block object is missing 'shape_name' attribute.")
                return []
            if not hasattr(current_block, 'rotation'):
                 print("Error: current_block object is missing 'rotation' attribute.")
                 return []
            if not hasattr(current_block, 'x') or not hasattr(current_block, 'y'):
                 print("Error: current_block object is missing 'x' or 'y' attribute.")
                 return []


            features_np = self._featurize_live_game_state(game_board_state, current_block, next_block)
            features_tensor = torch.tensor(features_np, dtype=torch.float).unsqueeze(0) 
            
            device = next(self.model.parameters()).device 
            features_tensor = features_tensor.to(device)

            with torch.no_grad():
                prediction_scores = self.model(features_tensor)
                _, predicted_label_idx = torch.max(prediction_scores, 1)
            
            predicted_action_name = self.label_to_action_const.get(predicted_label_idx.item())
            
            if predicted_action_name:
                pygame_key = self.action_name_to_pygame_key_const.get(predicted_action_name)
                if pygame_key is not None:
                    return [pygame_key] 
                else:
                    print(f"Warning: Predicted action name '{predicted_action_name}' has no Pygame key mapping.")
            else:
                print(f"Warning: Predicted label index {predicted_label_idx.item()} has no action name mapping.")

        except AttributeError as ae:
            print(f"AttributeError in NNAIPlayer get_next_move: {ae}. Check block object structure.")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"Error in NNAIPlayer get_next_move: {e}")
            import traceback
            traceback.print_exc()

        return [] 

    def update_game_state(self, game_board_state):
        pass 


# --- Main Execution ---
if __name__ == '__main__':
    print("--- Tetris AI: Feature Engineering & PyTorch NN ---")

    try:
        if not raw_data_points: # Check if raw_data_points is empty
             print("raw_data_points is empty. Cannot proceed with dataset creation. Exiting.")
             pygame.quit()
             exit()

        dataset = TetrisDataset(raw_data_points)
        if len(dataset) == 0:
            print("No data to train on after filtering. Exiting.")
            pygame.quit()
            exit()
            
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        if train_size == 0 or test_size == 0 or len(dataset) < 2 : 
             print(f"Dataset too small for train/test split (Total: {len(dataset)}). Using all for training and testing.")
             train_dataset = dataset
             test_dataset = dataset 
        else:
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        print(f"Dataset size: Total={len(dataset)}, Train={len(train_dataset)}, Test={len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=min(2, len(train_dataset)), shuffle=True) if len(train_dataset) > 0 else None
        test_loader = DataLoader(test_dataset, batch_size=min(2, len(test_dataset)), shuffle=False) if len(test_dataset) > 0 else None


        model = TetrisActionPredictor(INPUT_FEATURES_SIZE, NUM_ACTIONS)
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        if train_loader:
            num_training_epochs = 50 
            if len(train_dataset) < 5: 
                num_training_epochs = 20
            train_model(model, train_loader, criterion, optimizer, num_epochs=num_training_epochs)
            
            torch.save(model.state_dict(), MODEL_FILENAME)
            print(f"Trained model saved to {MODEL_FILENAME}")
        else:
            print("Skipping training as train_loader is empty.")


        if test_loader:
            evaluate_model(model, test_loader, criterion)
        else:
            print("Skipping evaluation as test_loader is empty or None.") # MODIFIED message
        
        print("\n--- Example Prediction for a single data point (using featurize_training_data) ---")
        if raw_data_points: # Check if raw_data_points is not empty
            user_sample_point_raw = raw_data_points[min(4, len(raw_data_points)-1)] # Avoid index error

            features_for_sample, actual_label_idx = featurize_training_data(user_sample_point_raw)
            
            if actual_label_idx != -1:
                features_tensor = torch.tensor(features_for_sample, dtype=torch.float).unsqueeze(0)
                device = next(model.parameters()).device
                features_tensor = features_tensor.to(device)
                
                model.eval()
                with torch.no_grad():
                    prediction_scores = model(features_tensor)
                    _, predicted_label_idx = torch.max(prediction_scores, 1)
                
                predicted_action_name = LABEL_TO_ACTION.get(predicted_label_idx.item(), "Unknown")
                actual_action_name = LABEL_TO_ACTION.get(actual_label_idx, "Unknown")
                
                print(f"Data Point Details (from raw_data_points index {min(4, len(raw_data_points)-1)}):")
                print(f"  Current Shape: {user_sample_point_raw['current_block_shape']}, Rotation: {user_sample_point_raw['current_block_rotation']}")
                print(f"  Actual Action: {actual_action_name}, Predicted Action: {predicted_action_name}")
            else:
                print(f"Could not make prediction for user sample as its action was unmapped: {user_sample_point_raw.get('action')}")
        else:
            print("raw_data_points is empty, cannot run example prediction.")


        print("\n--- Example Usage of NNAIPlayer ---")
        if os.path.exists(MODEL_FILENAME): 
            ai_player = NNAIPlayer(grid_width=BOARD_WIDTH, grid_height=BOARD_HEIGHT, model_path=MODEL_FILENAME)

            dummy_board_state = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
            dummy_board_state[18, 3:7] = 1 
            dummy_board_state[19, :] = 1   
            
            # Use the placeholder Block class defined in this script for the dummy objects
            dummy_current_block = Block(shape_name='TBlock', rotation_degrees=90, x_pos=4, y_pos=2)
            dummy_next_block = Block(shape_name='LBlock', rotation_degrees=0, x_pos=0, y_pos=0) 

            print("Querying NNAIPlayer for next move with dummy state:")
            predicted_actions_list = ai_player.get_next_move(dummy_board_state, dummy_current_block, dummy_next_block)
            
            if predicted_actions_list:
                predicted_pygame_key = predicted_actions_list[0]
                action_key_name = pygame.key.name(predicted_pygame_key) if predicted_pygame_key else "None"
                print(f"NNAIPlayer predicted action (Pygame Key): {predicted_pygame_key} ({action_key_name})")
            else:
                print("NNAIPlayer did not predict a valid action.")
        else:
            print(f"Skipping NNAIPlayer example usage as model file {MODEL_FILENAME} was not found (likely due to empty training set).")

    except ValueError as e:
        print(f"A ValueError occurred: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    pygame.quit() 
    print("\n--- Script Finished ---")

