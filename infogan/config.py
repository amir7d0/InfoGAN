
class Config(object):
    def __init__(self):
        self.epochs = 10
        self.batch_size = 128
        self.gen_lr = 1e-3
        self.dis_lr = 2e-4
        self.lambda_disc = 1.0
        self.lambda_cont = 0.1

        self.dataset_name = 'mnist'  # choose dataset from ['mnist', 'fashion_mnist', 'svhn', 'celeb_a']
        self.root_log_dir = "logs/"
        self.root_checkpoint_dir = "ckt/"

        self.wandb_api_key = ''  # add W&B api key


config = Config()
