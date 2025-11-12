# Model parameters
NUM_CLASSES = 1000
INPUT_CHANNELS = 3
INPUT_SIZE = 224

# ConvBlock parameters
CONV_BLOCKS = [
    (3, 64, 2),
    (64, 128, 2),
    (128, 256, 3),
    (256, 512, 3),
    (512, 512, 3)
]

# FC layer parameters
FC_LAYERS = [4096, 4096, NUM_CLASSES]

# Training hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 32
NUM_EPOCHS = 50
DROPOUT = 0.5

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
