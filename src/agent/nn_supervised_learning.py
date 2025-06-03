# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pygame",
#     "torch",
#     "torchvision",
#     "mlflow",
# ]
# ///

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pygame
import os
import json
import copy # For deep copying model state

from src.common.constants import BOARD_WIDTH_TILES, BOARD_HEIGHT_TILES

import mlflow
import mlflow.pytorch

# --- Constants and Configuration ---
pygame.init()

BOARD_WIDTH = BOARD_WIDTH_TILES
BOARD_HEIGHT = BOARD_HEIGHT_TILES

MODEL_DIR = "models"
MODEL_FILENAME = os.path.join(MODEL_DIR, "tetris_action_predictor.pth") # For the best model

MLFLOW_EXPERIMENT_NAME = "Tetris_Action_Prediction_Tuning"

ALL_SHAPES = ['IBlock', 'OBlock', 'TBlock', 'LBlock', 'ZBlock']
SHAPE_TO_IDX = {shape: i for i, shape in enumerate(ALL_SHAPES)}
NUM_UNIQUE_SHAPES = len(ALL_SHAPES)

POSSIBLE_ACTIONS = ["Left", "Right", "Rotate", "SoftDrop", "HardDrop"]
ACTION_TO_LABEL = {action: i for i, action in enumerate(POSSIBLE_ACTIONS)}
LABEL_TO_ACTION = {i: action for action, i in ACTION_TO_LABEL.items()}
NUM_ACTIONS = len(POSSIBLE_ACTIONS)

PYGAME_ACTION_MAP = {
    pygame.K_LEFT: "Left", pygame.K_RIGHT: "Right", pygame.K_UP: "Rotate",
    pygame.K_DOWN: "SoftDrop", pygame.K_SPACE: "HardDrop"
}
ACTION_NAME_TO_PYGAME_KEY = {v: k for k, v in PYGAME_ACTION_MAP.items()}

TRAINING_DATA_PATH = "data/tetris_training_data.json"

# --- Feature Engineering ---

def get_board_features(board_state_np):
    """
    Calculates various features from the game board state.

    Args:
        board_state_np (np.array): A 2D numpy array representing the game board,
                                   where 1 indicates a filled cell and 0 an empty cell.

    Returns:
        tuple: A tuple containing:
            - norm_column_heights (np.array): Normalized heights of each column.
            - norm_aggregate_height (float): Normalized sum of all column heights.
            - norm_num_holes (float): Normalized number of holes in the board.
            - norm_bumpiness (float): Normalized sum of height differences between adjacent columns.
            - norm_completed_lines (float): Normalized number of completed lines.
    """
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
    """
    Converts a single raw data point from collected data into a flat feature vector and a label.

    Args:
        data_point (dict): A dictionary representing a single data entry,
                           containing board state, block information, action, and score.

    Returns:
        tuple: A tuple containing:
            - features (np.array): The featurized numpy array for the model input.
            - label (int): The numerical label for the action taken, or -1 if unmapped.
    """
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
    rotation_norm = float(data_point.get('current_block_rotation', 0)) / 360.0
    pos_x_norm = 0.0
    pos_y_norm = 0.0
    if data_point.get('current_block_pos'):
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
    """
    PyTorch Dataset for Tetris training data.
    It takes raw data points, featurizes them, and provides an interface
    for DataLoader to get items.
    """
    def __init__(self, data_points):
        """
        Initializes the dataset.

        Args:
            data_points (list): A list of raw data point dictionaries.
        """
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
        """Returns the total number of samples in the dataset."""
        return len(self.features_list)
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the feature tensor and the label tensor.
        """
        return torch.tensor(self.features_list[idx], dtype=torch.float), \
               torch.tensor(self.labels_list[idx], dtype=torch.long)

# --- Input Feature Size Determination ---
def determine_input_feature_size(data_path=TRAINING_DATA_PATH):
    """
    Determines the size of the input feature vector by featurizing a sample data point.

    Args:
        data_path (str): Path to the training data JSON file.

    Returns:
        int: The calculated size of the input feature vector.
             Returns a fallback calculated size if data loading or featurization fails.
    """
    try:
        with open(data_path, 'r') as f:
            raw_data_points_for_size_calc = json.load(f)
        if not raw_data_points_for_size_calc:
            raise ValueError(f"Training data file '{data_path}' is empty.")
        valid_sample_found = False
        for dp_sample in raw_data_points_for_size_calc:
            sample_features_for_size_calc, sample_label = featurize_training_data(dp_sample)
            if sample_label != -1:
                return len(sample_features_for_size_calc)
        if not valid_sample_found:
            raise ValueError("No valid data points in sample to determine feature size.")
    except Exception as e:
        print(f"Warning/Error determining feature size: {e}. Falling back to calculated size.")
        return BOARD_WIDTH + 4 + NUM_UNIQUE_SHAPES + 3 + NUM_UNIQUE_SHAPES

INPUT_FEATURES_SIZE = determine_input_feature_size()

# --- PyTorch Model ---
class TetrisActionPredictor(nn.Module):
    """
    Neural network model for predicting Tetris actions.
    Consists of a sequence of linear layers with ReLU activations and Dropout.
    """
    def __init__(self, input_size, num_actions_out, model_arch_params):
        """
        Initializes the neural network.

        Args:
            input_size (int): The size of the input feature vector.
            num_actions_out (int): The number of possible output actions (size of output layer).
            model_arch_params (dict): Dictionary containing architecture parameters like
                                      output units of fully connected layers and dropout rates.
        """
        super(TetrisActionPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, model_arch_params["fc1_out"]), nn.ReLU(),
            nn.Dropout(model_arch_params["dropout1"]),
            nn.Linear(model_arch_params["fc1_out"], model_arch_params["fc2_out"]), nn.ReLU(),
            nn.Dropout(model_arch_params["dropout2"]),
            nn.Linear(model_arch_params["fc2_out"], model_arch_params["fc3_out"]), nn.ReLU(),
            nn.Dropout(model_arch_params["dropout3"]),
            nn.Linear(model_arch_params["fc3_out"], num_actions_out)
        )
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor (logits for each action).
        """
        return self.network(x)

# --- Training and Evaluation ---
def train_model_internal(model, dataloader, criterion, optimizer, num_epochs, current_run_id):
    """
    Internal function to train the PyTorch model. Logs metrics to MLflow.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        num_epochs (int): The number of epochs to train for.
        current_run_id (str): The MLflow run ID for logging.
    """
    print(f"\n--- Starting Training (MLflow Run: {current_run_id}) ---")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0; correct_predictions = 0; total_predictions = 0
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
        avg_epoch_loss = epoch_loss / len(dataloader) if dataloader and len(dataloader) > 0 else 0
        accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        mlflow.log_metric(f"epoch_train_loss", avg_epoch_loss, step=epoch)
        mlflow.log_metric(f"epoch_train_accuracy", accuracy, step=epoch)
    print("--- Training Finished ---")

def evaluate_model_internal(model, dataloader, criterion, current_run_id, prefix="eval"):
    """
    Internal function to evaluate the PyTorch model. Logs metrics to MLflow.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation/test data.
        criterion (torch.nn.Module): The loss function.
        current_run_id (str): The MLflow run ID for logging.
        prefix (str): Prefix for logged metrics (e.g., "eval", "test").

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average loss on the evaluation data.
            - accuracy (float): The accuracy on the evaluation data.
    """
    print(f"\n--- Starting Evaluation (MLflow Run: {current_run_id}) ---")
    model.eval()
    total_loss = 0.0; correct_predictions = 0; total_predictions = 0
    if dataloader is None or len(dataloader) == 0 :
        print(f"{prefix} dataloader empty. Skipping.")
        return 0.0, 0.0
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
    avg_loss = total_loss / len(dataloader) if dataloader and len(dataloader) > 0 else 0
    accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
    print(f"{prefix.capitalize()} Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    mlflow.log_metric(f"{prefix}_loss", avg_loss)
    mlflow.log_metric(f"{prefix}_accuracy", accuracy)
    print(f"--- {prefix.capitalize()} Finished ---")
    return avg_loss, accuracy

# --- NNAIPlayer Class ---
class NNAIPlayer:
    """
    An AI player that uses a trained neural network to decide the next move.
    Loads a pre-trained model from a local file path.
    """
    def __init__(self, grid_width, grid_height, model_path=MODEL_FILENAME):
        """
        Initializes the NNAIPlayer.

        Args:
            grid_width (int): The width of the game grid in tiles.
            grid_height (int): The height of the game grid in tiles.
            model_path (str): Path to the saved PyTorch model state dictionary.
        """
        self.grid_width = grid_width; self.grid_height = grid_height
        self.board_width_const = BOARD_WIDTH; self.board_height_const = BOARD_HEIGHT
        self.all_shapes_const = ALL_SHAPES; self.shape_to_idx_const = SHAPE_TO_IDX
        self.num_unique_shapes_const = NUM_UNIQUE_SHAPES
        self.label_to_action_const = LABEL_TO_ACTION
        self.action_name_to_pygame_key_const = ACTION_NAME_TO_PYGAME_KEY
        self.input_features_size = INPUT_FEATURES_SIZE # Global, determined once
        self.num_actions = NUM_ACTIONS
        # Note: For NNAIPlayer, MODEL_ARCHITECTURE_PARAMS must match the loaded model.
        # This example assumes a fixed architecture for the model being loaded by NNAIPlayer,
        # or that the architecture used for saving the best model is known.
        # For simplicity, we use default_arch_params if loading a generically saved model.
        default_arch_params = {"fc1_out": 512, "dropout1": 0.3, "fc2_out": 256, "dropout2": 0.3, "fc3_out": 128, "dropout3": 0.2}
        self.model = TetrisActionPredictor(self.input_features_size, self.num_actions, default_arch_params)

        if os.path.exists(model_path):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.to(device); self.model.eval()
                print(f"NNAIPlayer loaded model from {model_path} to {device}.")
            except Exception as e: print(f"Error loading model for NNAIPlayer: {e}. Player untrained.")
        else: print(f"Warning: Model {model_path} not found for NNAIPlayer. Player untrained.")
        print(f"NNAIPlayer init for {grid_width}x{grid_height} grid. Expects Features: {self.input_features_size}")

    def _featurize_live_game_state(self, game_board_state_np, current_block_obj, next_block_obj):
        """
        Converts the live game state into a feature vector for the NN model.
        Internal helper method.

        Args:
            game_board_state_np (np.array): Current state of the game board.
            current_block_obj (Block): The current falling block object.
            next_block_obj (Block): The next block object.

        Returns:
            np.array: The featurized numpy array.
        """
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
        """
        Determines the next move for the AI using the loaded neural network.

        Args:
            game_board_state (np.array): Current state of the game board.
            current_block (Block): The current falling block object.
            next_block (Block): The next block object.

        Returns:
            list: A list containing a single Pygame key for the predicted action,
                  or an empty list if no action is predicted or an error occurs.
        """
        try:
            attrs = ['shape_name', 'rotation', 'x', 'y'] # Required attributes for current_block
            for attr in attrs:
                if not hasattr(current_block, attr):
                    print(f"Error: current_block missing '{attr}'")
                    return []
            if not hasattr(next_block, 'shape_name'):
                print(f"Error: next_block missing 'shape_name'")
                return []

            features_np = self._featurize_live_game_state(game_board_state, current_block, next_block)
            if len(features_np) != self.input_features_size:
                print(f"Feature size mismatch: expected {self.input_features_size}, got {len(features_np)}")
                return []

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
            # import traceback; traceback.print_exc() # For more detailed debugging
        return []

    def update_game_state(self, game_board_state):
        """ Placeholder for potential future use, e.g., reinforcement learning. """
        pass

# --- Main Execution for Hyperparameter Tuning and Training ---
if __name__ == '__main__':
    print("--- Tetris AI: NN Hyperparameter Tuning with MLflow ---")

    if not os.path.exists(MODEL_DIR):
        try: os.makedirs(MODEL_DIR); print(f"Created model directory: {MODEL_DIR}")
        except OSError as e: print(f"Error creating model dir {MODEL_DIR}: {e}")

    raw_data_points = []
    try:
        with open(TRAINING_DATA_PATH, 'r') as f: raw_data_points = json.load(f)
        print(f"Loaded {len(raw_data_points)} data points from {TRAINING_DATA_PATH}")
    except Exception as e:
        print(f"Error loading data from {TRAINING_DATA_PATH}: {e}. Exiting.")
        pygame.quit(); exit()

    if not raw_data_points:
        print("No raw data points. Exiting."); pygame.quit(); exit()

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

    # Define hyperparameter search space
    hyperparameter_configs = [
        {"lr": 0.001, "batch_size": 32, "epochs": 30, "arch": {"fc1_out": 256, "dropout1": 0.3, "fc2_out": 128, "dropout2": 0.3, "fc3_out": 64, "dropout3": 0.2}},
        {"lr": 0.001, "batch_size": 64, "epochs": 50, "arch": {"fc1_out": 512, "dropout1": 0.3, "fc2_out": 256, "dropout2": 0.3, "fc3_out": 128, "dropout3": 0.2}},
        {"lr": 0.0005, "batch_size": 32, "epochs": 50, "arch": {"fc1_out": 512, "dropout1": 0.2, "fc2_out": 256, "dropout2": 0.2, "fc3_out": 128, "dropout3": 0.1}},
    ]

    best_eval_accuracy = -1.0
    best_model_state_dict = None
    best_hyperparams = None
    best_run_id = None

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    for i, hp_config in enumerate(hyperparameter_configs):
        print(f"\n--- Starting Hyperparameter Trial {i+1}/{len(hyperparameter_configs)} ---")
        print(f"Config: {hp_config}")

        with mlflow.start_run(run_name=f"Trial_{i+1}") as run:
            current_run_id = run.info.run_id
            print(f"MLflow Run ID: {current_run_id}")
            mlflow.log_param("mlflow_run_id", current_run_id)
            mlflow.log_params(hp_config) # Log all HP for this run
            mlflow.log_params({ # Log fixed params
                "board_width": BOARD_WIDTH, "board_height": BOARD_HEIGHT,
                "input_features_size": INPUT_FEATURES_SIZE,
                "num_unique_shapes": NUM_UNIQUE_SHAPES, "all_shapes": str(ALL_SHAPES),
                "num_actions": NUM_ACTIONS, "possible_actions": str(POSSIBLE_ACTIONS),
                "optimizer_type": "Adam", "total_dataset_size": len(dataset),
                "train_dataset_size": len(train_dataset), "test_dataset_size": len(test_dataset),
                "training_data_path": TRAINING_DATA_PATH
            })
            for key, value in hp_config["arch"].items(): mlflow.log_param(f"model_{key}", value)


            try:
                current_batch_size = hp_config["batch_size"]
                # Ensure batch size is not larger than dataset
                effective_train_batch_size = min(current_batch_size, len(train_dataset)) if len(train_dataset) > 0 else 1
                effective_test_batch_size = min(current_batch_size, len(test_dataset)) if len(test_dataset) > 0 else 1

                train_loader = DataLoader(train_dataset, batch_size=effective_train_batch_size, shuffle=True) if len(train_dataset) > 0 else None
                test_loader = DataLoader(test_dataset, batch_size=effective_test_batch_size, shuffle=False) if len(test_dataset) > 0 else None

                model = TetrisActionPredictor(INPUT_FEATURES_SIZE, NUM_ACTIONS, hp_config["arch"])
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=hp_config["lr"])

                if train_loader:
                    train_model_internal(model, train_loader, criterion, optimizer, hp_config["epochs"], current_run_id)
                    mlflow.pytorch.log_model(model, "tetris-action-predictor-model") # Log model for this run
                else:
                    mlflow.log_param("training_skipped", True)

                eval_loss, eval_accuracy = 0.0, 0.0
                if test_loader:
                    eval_loss, eval_accuracy = evaluate_model_internal(model, test_loader, criterion, current_run_id, prefix="eval")
                else:
                    mlflow.log_param("evaluation_skipped", True)

                if eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy
                    best_model_state_dict = copy.deepcopy(model.state_dict()) # Store copy of best weights
                    best_hyperparams = hp_config
                    best_run_id = current_run_id
                    print(f"*** New best model found in Trial {i+1} (Run ID: {current_run_id}) with Accuracy: {eval_accuracy:.2f}% ***")

                mlflow.set_tag("trial_status", "completed")

            except Exception as e:
                print(f"Error during MLflow run {current_run_id}: {e}")
                import traceback; traceback.print_exc()
                mlflow.set_tag("trial_status", "failed")
                mlflow.log_param("error_message", str(e))

    print("\n--- Hyperparameter Tuning Finished ---")
    if best_model_state_dict:
        print(f"Best Model (Run ID: {best_run_id}):")
        print(f"  Hyperparameters: {best_hyperparams}")
        print(f"  Evaluation Accuracy: {best_eval_accuracy:.2f}%")
        torch.save(best_model_state_dict, MODEL_FILENAME)
        print(f"Best model saved locally to {MODEL_FILENAME}")

        # Optionally, you can tag the best run in MLflow
        with mlflow.start_run(run_id=best_run_id, nested=False): # Re-open the best run to add a tag
             mlflow.set_tag("is_best_model", "true")
    else:
        print("No successful training run completed, so no best model to save.")

    pygame.quit()
    print("\n--- Script Finished ---")