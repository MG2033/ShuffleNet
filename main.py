from utils import parse_args, create_experiment_dirs, calculate_flops, show_parameters
from model import ShuffleNet
from train import Train
from data_loader import DataLoader
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
    # The batch size is equal to 1 when testing to simulate the real experiment.
    data_batch_size = config_args.batch_size if config_args.train_or_test == "train" else 1
    data = DataLoader(data_batch_size, config_args.shuffle)
    print("Loading Data...")
    config_args.img_height, config_args.img_width, config_args.num_channels, \
    config_args.train_data_size, config_args.test_data_size = data.load_data()
    print("Data loaded\n\n")

    # Model creation
    print("Building the model...")
    model = ShuffleNet(config_args)
    print("Model is built successfully\n\n")

    # Parameters visualization
    show_parameters()

    # Summarizer creation
    summarizer = Summarizer(sess, config_args.summary_dir)
    # Train class
    trainer = Train(sess, model, data, summarizer)

    if config_args.train_or_test == 'train':
        try:
            # print("FLOPs for batch size = " + str(config_args.batch_size) + "\n")
            # calculate_flops()
            print("Training...")
            trainer.train()
            print("Training Finished\n\n")
        except KeyboardInterrupt:
            trainer.save_model()

    elif config_args.train_or_test == 'test':
        # print("FLOPs for single inference \n")
        # calculate_flops()
        # This can be 'val' or 'test' or even 'train' according to the needs.
        print("Testing...")
        trainer.test('val')
        print("Testing Finished\n\n")

    else:
        raise ValueError("Train or Test options only are allowed")


if __name__ == '__main__':
    main()
