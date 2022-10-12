import os
import dotenv

dotenv.load_dotenv()



DATASET_PATH = os.environ.get('DATASET_PATH')
if DATASET_PATH is None: 
    DATASET_PATH = './data'
    os.makedirs(DATASET_PATH, exist_ok=True)


class Config:
    def __init__(self):
        self.dataset_path = DATASET_PATH

        self.checkpoint_dir = "model_cp"
        self.checkpoint_file_template = "cwavegan_{}.pt"
        self.checkpoint_interval = 5

        # Hyperparamters
        self.batch_size = 64
        self.epochs = 100
        self.num_workers = 8

        self.audio_length=16384
        self.latent_size=100
        self.model_size=64
        self.audio_channel = 1 # mono channel
        self.phase_shift_factor = 2
        
        self.g_lr = 1e-4
        self.d_lr = 4e-4  # instead n_critic different learning rate x3~5
        self.adam_betas = (0.5, 0.9)



