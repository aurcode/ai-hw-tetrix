import numpy as np
import random
import pygame

class NetMatrix:
    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols
        if data is not None:
            self.data = np.array(data)
        else:
            self.data = np.zeros((rows, cols))

    def randomize(self):
        """Randomize the matrix values between -1 and 1."""
        self.data = np.random.rand(self.rows, self.cols) * 2 - 1

    @staticmethod
    def single_column_net_matrix_from_array(arr):
        """Create a single column matrix from a 1D array."""
        return NetMatrix(len(arr), 1, data=arr)

    def to_array(self):
        """Convert the matrix back to a 1D list."""
        return self.data.flatten().tolist()

    def add_bias(self):
        """Add a bias term (a row of 1s) to the matrix."""
        # In this context, it appears the bias is added as an extra input node,
        # so we'll append a 1 to the input array.
        new_matrix = NetMatrix(self.rows + 1, 1)
        new_matrix.data[:self.rows, :] = self.data
        new_matrix.data[self.rows, 0] = 1
        return new_matrix

    def dot(self, other_matrix):
        """Perform dot product between this matrix and another."""
        result = np.dot(self.data, other_matrix.data)
        return NetMatrix(result.shape[0], result.shape[1], data=result)

    def activate(self):
        """Apply the activation function (sigmoid in this example)."""
        # A common choice for activation is the sigmoid function.
        activated_data = 1 / (1 + np.exp(-self.data))
        return NetMatrix(self.rows, self.cols, data=activated_data)

    def mutate(self, mutation_rate):
        """Mutate the matrix weights based on a mutation rate."""
        for i in range(self.rows):
            for j in range(self.cols):
                if random.random() < mutation_rate:
                    self.data[i, j] += np.random.normal(0, 0.1) # Add a small random value

    def crossover(self, partner):
        """Perform crossover with a partner matrix."""
        child = NetMatrix(self.rows, self.cols)
        # Randomly pick a crossover point
        rand_r = random.randint(0, self.rows -1)
        rand_c = random.randint(0, self.cols - 1)
        for i in range(self.rows):
            for j in range(self.cols):
                if i < rand_r or (i == rand_r and j <= rand_c):
                    child.data[i,j] = self.data[i,j]
                else:
                    child.data[i,j] = partner.data[i,j]
        return child

    def clone(self):
        """Create a deep copy of the matrix."""
        return NetMatrix(self.rows, self.cols, data=np.copy(self.data))


class NeuralNet:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, hidden_layers):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.h_layers = hidden_layers

        self.weights = [None] * (self.h_layers + 1)
        self.weights[0] = NetMatrix(self.h_nodes, self.i_nodes + 1)
        for i in range(1, self.h_layers):
            self.weights[i] = NetMatrix(self.h_nodes, self.h_nodes + 1)
        self.weights[-1] = NetMatrix(self.o_nodes, self.h_nodes + 1)

        for w in self.weights:
            w.randomize()

    def output(self, inputs_arr):
        """Calculate the output of the neural network."""
        inputs = NetMatrix.single_column_net_matrix_from_array(inputs_arr)
        curr_bias = inputs.add_bias()

        for i in range(self.h_layers):
            hidden_ip = self.weights[i].dot(curr_bias)
            hidden_op = hidden_ip.activate()
            curr_bias = hidden_op.add_bias()

        output_ip = self.weights[-1].dot(curr_bias)
        # The original code doesn't show an activation on the final output,
        # which might be intentional (e.g., for regression).
        # If activation is needed, uncomment the next line.
        # output = output_ip.activate()
        output = output_ip
        
        return output.to_array()

    def mutate(self, mr):
        """Mutate all weight matrices."""
        for w in self.weights:
            w.mutate(mr)

    def crossover(self, partner):
        """Create a child network by crossing over weights with a partner."""
        child = NeuralNet(self.i_nodes, self.h_nodes, self.o_nodes, self.h_layers)
        for i in range(len(self.weights)):
            child.weights[i] = self.weights[i].crossover(partner.weights[i])
        return child

    def clone(self):
        """Create a deep copy of the neural network."""
        clone = NeuralNet(self.i_nodes, self.h_nodes, self.o_nodes, self.h_layers)
        for i in range(len(self.weights)):
            clone.weights[i] = self.weights[i].clone()
        return clone

    def load(self, loaded_weights):
        """Load weights from another source."""
        for i in range(len(self.weights)):
            self.weights[i] = loaded_weights[i]

    def pull(self):
        """Return a copy of the network's weights."""
        return [w.clone() for w in self.weights]

    def show(self, screen, x, y, w, h, vision, decision):
        """Draw the neural network on a pygame screen."""
        space = 5
        n_size = (h - (space * (self.i_nodes - 2))) / self.i_nodes if self.i_nodes > 1 else h
        n_space = (w - (len(self.weights) * n_size)) / len(self.weights)
        h_buff = (h - (space * (self.h_nodes - 1)) - (n_size * self.h_nodes)) / 2
        o_buff = (h - (space * (self.o_nodes - 1)) - (n_size * self.o_nodes)) / 2
        
        # Draw weights
        lc = 1
        # Input to Hidden
        for i in range(self.weights[0].rows):
            for j in range(self.weights[0].cols - 1):
                color = (0, 0, 255) if self.weights[0].data[i][j] >= 0 else (255, 0, 0)
                start_pos = (x + n_size, y + (n_size / 2) + (j * (space + n_size)))
                end_pos = (x + n_size + n_space, y + h_buff + (n_size / 2) + (i * (space + n_size)))
                pygame.draw.line(screen, color, start_pos, end_pos, 2)
        
        lc += 1
        # Hidden to Hidden
        for a in range(1, self.h_layers):
            for i in range(self.weights[a].rows):
                for j in range(self.weights[a].cols - 1):
                    color = (0, 0, 255) if self.weights[a].data[i][j] >= 0 else (255, 0, 0)
                    start_pos = (x + (lc * n_size) + ((lc - 1) * n_space), y + h_buff + (n_size / 2) + (j * (space + n_size)))
                    end_pos = (x + (lc * n_size) + (lc * n_space), y + h_buff + (n_size / 2) + (i * (space + n_size)))
                    pygame.draw.line(screen, color, start_pos, end_pos, 2)
            lc += 1

        # Hidden to Output
        for i in range(self.weights[-1].rows):
            for j in range(self.weights[-1].cols - 1):
                color = (0, 0, 255) if self.weights[-1].data[i][j] >= 0 else (255, 0, 0)
                start_pos = (x + (lc * n_size) + ((lc-1) * n_space), y + h_buff + (n_size / 2) + (j * (space + n_size)))
                end_pos = (x + (lc * n_size) + (lc * n_space), y + o_buff + (n_size / 2) + (i * (space + n_size)))
                pygame.draw.line(screen, color, start_pos, end_pos, 2)

        # Draw nodes
        lc = 0
        font = pygame.font.SysFont(None, int(n_size/2))
        
        # Input nodes
        for i in range(self.i_nodes):
            color = (0, 255, 0) if vision[i] != 0 else (255, 255, 255)
            pygame.draw.ellipse(screen, color, (x, y + (i * (n_size + space)), n_size, n_size))
            pygame.draw.ellipse(screen, (0,0,0), (x, y + (i * (n_size + space)), n_size, n_size), 1)
            img = font.render(str(i), True, (0,0,0))
            screen.blit(img, (x + n_size/2 - img.get_width()/2, y + (i * (n_size + space)) + n_size/2 - img.get_height()/2))
        
        lc += 1
        # Hidden nodes
        for a in range(self.h_layers):
            for i in range(self.h_nodes):
                pygame.draw.ellipse(screen, (255,255,255), (x + (lc * n_size) + (lc * n_space), y + h_buff + (i * (n_size + space)), n_size, n_size))
                pygame.draw.ellipse(screen, (0,0,0), (x + (lc * n_size) + (lc * n_space), y + h_buff + (i * (n_size + space)), n_size, n_size), 1)
            lc += 1

        # Output nodes
        for i in range(self.o_nodes):
            pygame.draw.ellipse(screen, (255,255,255), (x + (lc * n_space) + (lc * n_size), y + o_buff + (i * (n_size + space)), n_size, n_size))
            pygame.draw.ellipse(screen, (0,0,0), (x + (lc * n_space) + (lc * n_size), y + o_buff + (i * (n_size + space)), n_size, n_size), 1)
        
        # Score text
        score_font = pygame.font.SysFont(None, 15)
        img = score_font.render("Score", True, (0,0,0))
        screen.blit(img, (x + (lc * n_space) + (lc * n_size) + n_size/2 - img.get_width()/2, y + o_buff + n_size/2 - 20))


if __name__ == '__main__':
    # --- Example Usage ---
    pygame.init()

    # Screen dimensions
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Neural Network Visualization")
    
    # Create a neural network
    # Parameters: input_nodes, hidden_nodes, output_nodes, hidden_layers
    nn = NeuralNet(5, 8, 2, 2)
    
    # Example input and output data
    vision_data = [1, 0, 1, 0, 1] 
    decision_data = nn.output(vision_data)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background
        screen.fill((200, 200, 200)) # A light gray background

        # Display the neural network
        # Parameters: screen, x, y, width, height, vision_data, decision_data
        nn.show(screen, 50, 50, 700, 500, vision_data, decision_data)

        # Update the display
        pygame.display.flip()

    pygame.quit()