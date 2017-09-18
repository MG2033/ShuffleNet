from utils import parse_args, create_experiment_dirs
from model import ShuffleNet
from train import Train
from data import DataLoader
from summarizer import Summarizer
import tensorflow as tf


def main():
    # Parse the JSON arguments
    config_args = parse_args()

    # Create the experiment directories
    _, config_args.summary_dir, config_args.checkpoint_dir = create_experiment_dirs(config_args.experiment_dir)

    # Reset the default Tensorflow graph
    tf.reset_default_graph()

    # Tensorflow specific configuration
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Data loading
    data = DataLoader(config_args.batch_size, config_args.shuffle)
    config_args.img_height, config_args.img_width, config_args.num_channels, \
    config_args.train_data_size, config_args.test_data_size = data.load_data()

    # Model creation
    model = ShuffleNet(config_args)

    # Summarizer creation
    summarizer = Summarizer(sess, config_args.summary_dir)

    # Train class
    trainer = Train(sess, model, data, summarizer)

    print("Training...")
    trainer.train()
    print("Training Finished")

    print("Final test!")
    trainer.test()
    print("Testing Finished")


if __name__ == '__main__':
    main()
