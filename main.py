from utils import parse_args, create_experiment_dirs
from model import ShuffleNet
from train import Train
from data import DataLoader
from summarizer import Summarizer
import tensorflow as tf


def main():
    # Parse the JSON arguments
    try:
        config_args = parse_args()
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

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
    print("Loading Data...")
    config_args.img_height, config_args.img_width, config_args.num_channels, \
    config_args.train_data_size, config_args.test_data_size = data.load_data()
    print("Data loaded\n\n")

    # Model creation
    print("Building the model...")
    model = ShuffleNet(config_args)
    print("Model is built successfully\n\n")

    # Summarizer creation
    summarizer = Summarizer(sess, config_args.summary_dir)

    # Train class
    trainer = Train(sess, model, data, summarizer)

    try:
        print("Training...")
        trainer.train()
        print("Training Finished\n\n")
    except:
        trainer.save_model()

    print("Final test!")
    # trainer.test()
    print("Testing Finished\n\n")


if __name__ == '__main__':
    main()
