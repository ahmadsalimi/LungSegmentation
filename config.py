class Config:
    def __init__(self):
        self.data_root_path = "####"
        self.model_store_directory = "####"
        self.log_directory = "####"
        self.epochs = 100
        self.batch_size = 1
        self.batch_norms = (
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True
        )
        self.learning_rate = 1e-4
        self.sample_per_epoch = 640

config = Config()