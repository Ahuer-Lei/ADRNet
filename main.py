import os
from options import TrainOptions
from train import Train_and_test

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():

    parser = TrainOptions()   
    config = parser.parse()

    print('\n--- options load success ---')

    trainer = Train_and_test(config)
    if config.train:
        trainer.train_and_test()
    else:
        trainer.test()


if __name__ == '__main__':
    main()