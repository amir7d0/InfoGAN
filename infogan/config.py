
class Config:
    def __init__(self):
        self.epochs = 10
        self.batch_size = 128
        self.gen_lr = 1e-3
        self.dis_lr = 2e-4

        self.name = 'mnist'
        self.root_log_dir = "logs/"
        self.root_checkpoint_dir = "ckt/"


config = Config()