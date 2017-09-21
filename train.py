import tensorflow as tf
from tqdm import tqdm
import numpy as np


class Train:
    """Trainer class for the CNN.
    It's also responsible for loading/saving the model checkpoints from/to experiments/experiment_name/checkpoint_dir"""

    def __init__(self, sess, model, data, summarizer):
        self.sess = sess
        self.model = model
        self.args = self.model.args
        self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep,
                                    keep_checkpoint_every_n_hours=10,
                                    save_relative_paths=True)
        # Summarizer references
        self.data = data
        self.summarizer = summarizer

        # Initializing the model
        self.init = None
        self.__init_model()

        # Loading the model checkpoint if exists
        self.__load_model()

    ############################################################################################################
    # Model related methods
    def __init_model(self):
        print("Initializing the model...")
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        print("Model initialized\n\n")

    def save_model(self):
        """
        Save Model Checkpoint
        :return:
        """
        print("Saving a checkpoint")
        self.saver.save(self.sess, self.args.checkpoint_dir, self.model.global_step_tensor)
        print("Checkpoint Saved\n\n")

    def __load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Checkpoint loaded\n\n")
        else:
            print("First time to train!\n\n")

    ############################################################################################################
    # Train and Test methods
    def train(self):
        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, self.args.num_epochs + 1, 1):

            # Initialize tqdm
            num_iterations = self.args.train_data_size // self.args.batch_size
            tqdm_batch = tqdm(self.data.generate_batch(type='train'), total=num_iterations,
                              desc="Epoch-" + str(cur_epoch) + "-")

            # Initialize the current iterations
            cur_iteration = 0

            # Initialize classification accuracy and loss lists
            loss_list = []
            acc_list = []

            # Loop by the number of iterations
            for X_batch, y_batch in tqdm_batch:
                # Get the current iteration for summarizing it
                cur_it = self.model.global_step_tensor.eval(self.sess)

                # Feed this variables to the network
                feed_dict = {self.model.X: X_batch,
                             self.model.y: y_batch,
                             self.model.is_training: True
                             }
                # Run the feed_forward
                _, loss, acc = self.sess.run(
                    [self.model.train_op, self.model.loss, self.model.accuracy],
                    feed_dict=feed_dict)
                # Append loss and accuracy
                loss_list += [loss]
                acc_list += [acc]

                # Update the Global step
                self.model.global_step_assign_op.eval(session=self.sess,
                                                      feed_dict={self.model.global_step_input: cur_it + 1})

                if cur_it >= num_iterations - 1:
                    avg_loss = np.mean(loss_list)
                    avg_acc = np.mean(acc_list)
                    # summarize
                    summaries_dict = dict()
                    summaries_dict['loss'] = avg_loss
                    summaries_dict['acc'] = avg_acc

                    # summarize
                    self.summarizer.add_summary(cur_it, summaries_dict=summaries_dict)

                    # Update the Current Epoch tensor
                    self.model.global_epoch_assign_op.eval(session=self.sess,
                                                           feed_dict={self.model.global_epoch_input: cur_epoch + 1})

                    # Print in console
                    tqdm_batch.close()
                    print("Epoch-" + str(cur_epoch) + " | " + "loss: " + str(avg_loss) + " -" + " acc: " + str(
                        avg_acc)[
                                                                                                           :7])
                    # Break the loop to finalize this epoch
                    break

                # Update the Global step (current iteration)
                self.model.global_step_assign_op.eval(session=self.sess,
                                                      feed_dict={self.model.global_step_input: cur_it + 1})

                # Update the current iteration
                cur_iteration += 1

            # Save the current checkpoint
            if cur_epoch % self.args.save_model_every == 0:
                self.save_model()

            # Test the model on validation or test data
            if cur_epoch % self.args.test_every == 0:
                self.test('val')

    def test(self, test_type='val'):
        num_iterations = self.args.test_data_size // self.args.batch_size
        tqdm_batch = tqdm(self.data.generate_batch(type=test_type), total=num_iterations,
                          desc='Testing')
        # Initialize classification accuracy and loss lists
        loss_list = []
        acc_list = []
        cur_iteration = 0

        for X_batch, y_batch in tqdm_batch:
            # Feed this variables to the network
            feed_dict = {self.model.X: X_batch,
                         self.model.y: y_batch,
                         self.model.is_training: False
                         }
            # Run the feed_forward
            loss, acc = self.sess.run(
                [self.model.loss, self.model.accuracy],
                feed_dict=feed_dict)
            # Append loss and accuracy
            loss_list += [loss]
            acc_list += [acc]
            cur_iteration += 1

            if cur_iteration >= num_iterations - 1:
                avg_loss = np.mean(loss_list)
                avg_acc = np.mean(acc_list)
                print('Test results | test_loss: ' + str(avg_loss) + ' - test_acc: ' + str(avg_acc)[:7])
                break
