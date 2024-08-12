import argparse


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()


    # data related
        self.parser.add_argument('--train_data_path', type=str, default='./train', help='path to train data')
        self.parser.add_argument('--test_data_path', type=str, default='./test', help='path to test data')
        self.parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
        self.parser.add_argument('--nThreads', type=int, default=8, help='cpu threads')
        self.parser.add_argument('--train', action='store_true',default=True, help='Train mode or test mode')
    
    # generate data
        self.parser.add_argument('--rotation', type=int, default=10)
        self.parser.add_argument('--translation', type=float, default=0.1)
        self.parser.add_argument('--scaling', type=float, default=0.12)
        self.parser.add_argument('--dim', type=int, default=2)

    # output related
        self.parser.add_argument('--train_img_dir', type=str, default='./results', help='save path for training process visualization')
        self.parser.add_argument('--model_dir', type=str, default='./model_save', help='save path for model weights')

    # training related
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='')
        self.parser.add_argument('--n_ep', type=int, default=300, help='epoch')
        self.parser.add_argument('--n_ep_decay', type=int, default=150, help='which epoch starts to reduce the learning rate，-1：不变')

    # test related
        self.parser.add_argument('--train_logs_dir', type=str, default='./logs/train', help='Save path for training information')
        self.parser.add_argument('--test_logs_dir', type=str, default='./logs/test', help='Save path for testing information')



    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- train and test settings ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
