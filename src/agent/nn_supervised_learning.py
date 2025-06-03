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
import json

# INTEGRATION: Import constants from the common module
from src.common.constants import BOARD_WIDTH_TILES, BOARD_HEIGHT_TILES

# MLFLOW INTEGRATION: Import mlflow
import mlflow
import mlflow.pytorch

# --- Constants and Configuration ---
pygame.init() # Initialize Pygame to use its key constants

BOARD_WIDTH = BOARD_WIDTH_TILES
BOARD_HEIGHT = BOARD_HEIGHT_TILES

MODEL_DIR = "models" # Local model saving directory
MODEL_FILENAME = os.path.join(MODEL_DIR, "tetris_action_predictor.pth") # Local model filename

# MLFLOW INTEGRATION: Experiment Name
MLFLOW_EXPERIMENT_NAME = "Tetris_Action_Prediction"

ALL_SHAPES = ['IBlock', 'OBlock', 'TBlock', 'LBlock', 'ZBlock']
SHAPE_TO_IDX = {shape: i for i, shape in enumerate(ALL_SHAPES)}
NUM_UNIQUE_SHAPES = len(ALL_SHAPES)

POSSIBLE_ACTIONS = ["Left", "Right", "Rotate", "SoftDrop", "HardDrop"]
ACTION_TO_LABEL = {action: i for i, action in enumerate(POSSIBLE_ACTIONS)}
LABEL_TO_ACTION = {i: action for action, i in ACTION_TO_LABEL.items()}
NUM_ACTIONS = len(POSSIBLE_ACTIONS)

PYGAME_ACTION_MAP = {
    pygame.K_LEFT: "Left",
    pygame.K_RIGHT: "Right",
    pygame.K_UP: "Rotate",
    pygame.K_DOWN: "SoftDrop",
    pygame.K_SPACE: "HardDrop"
}
ACTION_NAME_TO_PYGAME_KEY = {v: k for k, v in PYGAME_ACTION_MAP.items()}

# --- Feature Engineering Functions ---
def get_board_features(board_state_np):
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
    board_state_list = data_point['board_state']
    board_state_np = np.array(board_state_list, dtype=float)
    (norm_col_heights, norm_agg_h, norm_holes,
     norm_bump, norm_lines) = get_board_features(board_state_np)
    current_shape_one_hot = np.zeros(NUM_UNIQUE_SHAPES, dtype=float)
    current_shape_name = data_point['current_block_shape']
    if current_shape_name == 'LineBlock': current_shape_name = 'IBlock'
    if current_shape_name == 'SquareBlock': current_shape_name = 'OBlock'
    shape_idx = SHAPE_TO_IDX.get(current_shape_name)
    if shape_idx is not None:
        current_shape_one_hot[shape_idx] = 1.0
    rotation_norm = float(data_point['current_block_rotation']) / 360.0
    pos_x_norm = 0.0
    pos_y_norm = 0.0
    if data_point['current_block_pos']:
        pos_x_norm = float(data_point['current_block_pos'][0]) / BOARD_WIDTH
        pos_y_norm = float(data_point['current_block_pos'][1]) / BOARD_HEIGHT
    next_shape_one_hot = np.zeros(NUM_UNIQUE_SHAPES, dtype=float)
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
            features, label = featurize_training_data(dp)
            if label != -1:
                self.features_list.append(features)
                self.labels_list.append(label)
        if not self.features_list:
            raise ValueError("No valid data points found for training after featurization.")
    def __len__(self):
        return len(self.features_list)
    def __getitem__(self, idx):
        return torch.tensor(self.features_list[idx], dtype=torch.float), \
               torch.tensor(self.labels_list[idx], dtype=torch.long)

# --- Determine Input Feature Size ---
file_path = "data/tetris_training_data.json"
INPUT_FEATURES_SIZE = 0
try:
    with open(file_path, 'r') as f:
        raw_data_points_for_size_calc = json.load(f)
    if not raw_data_points_for_size_calc:
        raise ValueError(f"Training data file '{file_path}' is empty.")
    valid_sample_found = False
    for dp_sample in raw_data_points_for_size_calc:
        sample_features_for_size_calc, sample_label = featurize_training_data(dp_sample)
        if sample_label != -1:
            INPUT_FEATURES_SIZE = len(sample_features_for_size_calc)
            valid_sample_found = True
            break
    if not valid_sample_found:
        raise ValueError("No valid data points in sample to determine feature size.")
except Exception as e:
    INPUT_FEATURES_SIZE = BOARD_WIDTH + 4 + NUM_UNIQUE_SHAPES + 3 + NUM_UNIQUE_SHAPES
    print(f"Warning/Error determining feature size: {e}. Falling back to calculated: {INPUT_FEATURES_SIZE}")


# --- PyTorch Model ---
MODEL_ARCHITECTURE_PARAMS = { # For MLflow logging
    "fc1_out": 512, "dropout1": 0.3,
    "fc2_out": 256, "dropout2": 0.3,
    "fc3_out": 128, "dropout3": 0.2,
}
class TetrisActionPredictor(nn.Module):
    def __init__(self, input_size, num_actions_out):
        super(TetrisActionPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, MODEL_ARCHITECTURE_PARAMS["fc1_out"]), nn.ReLU(),
            nn.Dropout(MODEL_ARCHITECTURE_PARAMS["dropout1"]),
            nn.Linear(MODEL_ARCHITECTURE_PARAMS["fc1_out"], MODEL_ARCHITECTURE_PARAMS["fc2_out"]), nn.ReLU(),
            nn.Dropout(MODEL_ARCHITECTURE_PARAMS["dropout2"]),
            nn.Linear(MODEL_ARCHITECTURE_PARAMS["fc2_out"], MODEL_ARCHITECTURE_PARAMS["fc3_out"]), nn.ReLU(),
            nn.Dropout(MODEL_ARCHITECTURE_PARAMS["dropout3"]),
            nn.Linear(MODEL_ARCHITECTURE_PARAMS["fc3_out"], num_actions_out)
        )
    def forward(self, x):
        return self.network(x)

# --- Training and Evaluation ---
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    print(f"\n--- Starting Training (Input: {INPUT_FEATURES_SIZE}, Output: {NUM_ACTIONS}) ---")
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
        avg_epoch_loss = epoch_loss / len(dataloader) if dataloader else 0
        accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        # MLFLOW INTEGRATION: Log metrics per epoch
        mlflow.log_metric(f"epoch_train_loss", avg_epoch_loss, step=epoch)
        mlflow.log_metric(f"epoch_train_accuracy", accuracy, step=epoch)
    print("--- Training Finished ---")

def evaluate_model(model, dataloader, criterion, epoch_num=None): # Added epoch_num for step
    print("\n--- Starting Evaluation ---")
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    if dataloader is None or len(dataloader) == 0 :
        print("Evaluation dataloader empty. Skipping.")
        return 0.0, 0.0
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
    avg_loss = total_loss / len(dataloader) if dataloader else 0
    accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
    print(f"Evaluation Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    # MLFLOW INTEGRATION: Log evaluation metrics
    metric_prefix = "final_eval" if epoch_num is None else f"epoch_{epoch_num}_eval"
    mlflow.log_metric(f"{metric_prefix}_loss", avg_loss, step=epoch_num if epoch_num is not None else 0)
    mlflow.log_metric(f"{metric_prefix}_accuracy", accuracy, step=epoch_num if epoch_num is not None else 0)
    print("--- Evaluation Finished ---")
    return avg_loss, accuracy

# --- Placeholder Block Class ---
class PlaceholderBlock:
    def __init__(self, shape_name, rotation_degrees, x_pos, y_pos, points=None):
        self.shape_name = shape_name; self.rotation = rotation_degrees
        self.x = x_pos; self.y = y_pos; self.points = points if points is not None else []

# --- NNAIPlayer Class (remains largely the same for loading local model) ---
class NNAIPlayer:
    def __init__(self, grid_width, grid_height, model_path=MODEL_FILENAME):
        self.grid_width = grid_width; self.grid_height = grid_height
        self.board_width_const = BOARD_WIDTH; self.board_height_const = BOARD_HEIGHT
        self.all_shapes_const = ALL_SHAPES; self.shape_to_idx_const = SHAPE_TO_IDX
        self.num_unique_shapes_const = NUM_UNIQUE_SHAPES
        self.label_to_action_const = LABEL_TO_ACTION
        self.action_name_to_pygame_key_const = ACTION_NAME_TO_PYGAME_KEY
        self.input_features_size = INPUT_FEATURES_SIZE
        self.num_actions = NUM_ACTIONS
        self.model = TetrisActionPredictor(self.input_features_size, self.num_actions)
        if os.path.exists(model_path):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.to(device); self.model.eval()
                print(f"NNAIPlayer loaded model from {model_path} to {device}.")
            except Exception as e: print(f"Error loading model: {e}. NNAIPlayer untrained.")
        else: print(f"Warning: Model {model_path} not found. NNAIPlayer untrained.")
        print(f"NNAIPlayer init for {grid_width}x{grid_height} grid. Features: {self.input_features_size}")

    def _featurize_live_game_state(self, game_board_state_np, current_block_obj, next_block_obj):
        game_board_state_np = game_board_state_np.astype(float)
        (norm_col_heights, norm_agg_h, norm_holes,
         norm_bump, norm_lines) = get_board_features(game_board_state_np)
        current_shape_one_hot = np.zeros(self.num_unique_shapes_const, dtype=float)
        current_shape_name = current_block_obj.shape_name
        if current_shape_name == 'LineBlock': current_shape_name = 'IBlock'
        if current_shape_name == 'SquareBlock': current_shape_name = 'OBlock'
        shape_idx = self.shape_to_idx_const.get(current_shape_name)
        if shape_idx is not None: current_shape_one_hot[shape_idx] = 1.0
        rotation_norm = float(current_block_obj.rotation) / 360.0
        pos_x_norm = float(current_block_obj.x) / self.board_width_const
        pos_y_norm = float(current_block_obj.y) / self.board_height_const
        next_shape_one_hot = np.zeros(self.num_unique_shapes_const, dtype=float)
        next_shape_name = next_block_obj.shape_name
        if next_shape_name == 'LineBlock': next_shape_name = 'IBlock'
        if next_shape_name == 'SquareBlock': next_shape_name = 'OBlock'
        next_shape_idx = self.shape_to_idx_const.get(next_shape_name)
        if next_shape_idx is not None: next_shape_one_hot[next_shape_idx] = 1.0
        features = np.concatenate([
            norm_col_heights, np.array([norm_agg_h, norm_holes, norm_bump, norm_lines]),
            current_shape_one_hot, np.array([rotation_norm, pos_x_norm, pos_y_norm]),
            next_shape_one_hot
        ]).astype(np.float32)
        return features

    def get_next_move(self, game_board_state, current_block, next_block):
        try:
            attrs = ['shape_name', 'rotation', 'x', 'y']
            for attr in attrs:
                if not hasattr(current_block, attr): return []
            if not hasattr(next_block, 'shape_name'): return []
            features_np = self._featurize_live_game_state(game_board_state, current_block, next_block)
            if len(features_np) != self.input_features_size: return []
            features_tensor = torch.tensor(features_np, dtype=torch.float).unsqueeze(0)
            device = next(self.model.parameters()).device
            features_tensor = features_tensor.to(device)
            with torch.no_grad():
                prediction_scores = self.model(features_tensor)
                _, predicted_label_idx = torch.max(prediction_scores, 1)
            predicted_action_name = self.label_to_action_const.get(predicted_label_idx.item())
            if predicted_action_name:
                pygame_key = self.action_name_to_pygame_key_const.get(predicted_action_name)
                if pygame_key is not None: return [pygame_key]
        except Exception as e:
            print(f"Error in NNAIPlayer get_next_move: {e}")
        return []
    def update_game_state(self, game_board_state): pass

# --- Main Execution (for training the model) ---
if __name__ == '__main__':
    print("--- Tetris AI: Feature Engineering & PyTorch NN with MLflow ---")
    
    if not os.path.exists(MODEL_DIR):
        try: os.makedirs(MODEL_DIR); print(f"Created model directory: {MODEL_DIR}")
        except OSError as e: print(f"Error creating model dir {MODEL_DIR}: {e}")

    raw_data_points = []
    training_data_path = "data/tetris_training_data.json"
    try:
        with open(training_data_path, 'r') as f: raw_data_points = json.load(f)
        print(f"Loaded {len(raw_data_points)} data points from {training_data_path}")
    except Exception as e:
        print(f"Error loading data from {training_data_path}: {e}. Exiting.")
        pygame.quit(); exit()

    if not raw_data_points:
        print("No raw data points. Exiting."); pygame.quit(); exit()

    # MLFLOW INTEGRATION: Set experiment and start run
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_param("mlflow_run_id", run.info.run_id)

        try:
            dataset = TetrisDataset(raw_data_points)
            if len(dataset) == 0:
                print("No data to train on after filtering. Exiting."); pygame.quit(); exit()

            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            if train_size == 0 or test_size == 0 or len(dataset) < 2:
                train_dataset, test_dataset = dataset, dataset
            else:
                train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
            print(f"Dataset: Total={len(dataset)}, Train={len(train_dataset)}, Test={len(test_dataset)}")

            # MLFLOW INTEGRATION: Log dataset parameters
            mlflow.log_param("total_dataset_size", len(dataset))
            mlflow.log_param("train_dataset_size", len(train_dataset))
            mlflow.log_param("test_dataset_size", len(test_dataset))
            mlflow.log_param("training_data_path", training_data_path)


            batch_size = min(32, len(train_dataset)) if len(train_dataset) > 0 else 1
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if len(train_dataset) > 0 else None
            test_loader = DataLoader(test_dataset, batch_size=min(32, len(test_dataset)), shuffle=False) if len(test_dataset) > 0 else None
            
            model = TetrisActionPredictor(INPUT_FEATURES_SIZE, NUM_ACTIONS)
            criterion = nn.CrossEntropyLoss()
            learning_rate = 0.001
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            num_training_epochs = 50
            if len(train_dataset) < 20: num_training_epochs = max(5, len(train_dataset) * 2)

            # MLFLOW INTEGRATION: Log hyperparameters and model architecture
            mlflow.log_param("board_width", BOARD_WIDTH)
            mlflow.log_param("board_height", BOARD_HEIGHT)
            mlflow.log_param("input_features_size", INPUT_FEATURES_SIZE)
            mlflow.log_param("num_unique_shapes", NUM_UNIQUE_SHAPES)
            mlflow.log_param("all_shapes", str(ALL_SHAPES))
            mlflow.log_param("num_actions", NUM_ACTIONS)
            mlflow.log_param("possible_actions", str(POSSIBLE_ACTIONS))
            mlflow.log_param("epochs", num_training_epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("optimizer", "Adam")
            for key, value in MODEL_ARCHITECTURE_PARAMS.items():
                mlflow.log_param(f"model_{key}", value)
            
            # Log the script itself as an artifact
            try:
                mlflow.log_artifact(__file__)
            except Exception as e:
                print(f"Warning: Could not log script as artifact: {e}")


            if train_loader:
                train_model(model, train_loader, criterion, optimizer, num_epochs=num_training_epochs)
                # Save local model as before
                torch.save(model.state_dict(), MODEL_FILENAME)
                print(f"Trained model saved locally to {MODEL_FILENAME}")
                
                # MLFLOW INTEGRATION: Log the PyTorch model
                mlflow.pytorch.log_model(model, "tetris-action-predictor-model")
                print("Trained model logged to MLflow.")
            else:
                print("Skipping training as train_loader is empty.")
                mlflow.log_param("training_skipped", True)

            if test_loader:
                evaluate_model(model, test_loader, criterion) # Will log final eval metrics
            else:
                print("Skipping evaluation as test_loader is empty.")
                mlflow.log_param("evaluation_skipped", True)

            # ... (Example Prediction and NNAIPlayer Usage sections can remain as is) ...
            print("\n--- Example Prediction for a single data point ---")
            # (This section can stay for local script testing)

            print("\n--- Example Usage of NNAIPlayer (using local model) ---")
            # (This section remains, NNAIPlayer still loads from MODEL_FILENAME by default)


        except Exception as e:
            print(f"An error occurred during the MLflow run: {e}")
            import traceback
            traceback.print_exc()
            # MLFLOW INTEGRATION: Log error to MLflow if possible
            mlflow.set_tag("run_status", "failed")
            mlflow.log_param("error_message", str(e))

    pygame.quit()
    print("\n--- Script Finished ---")