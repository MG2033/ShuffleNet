import tensorflow as tf
from tqdm import tqdm


class Train:
    def __init__(self, sess, model):
        self.sess = sess
        self.model = model
        self.args = self.model.args
        self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep,
                                    keep_checkpoint_every_n_hours=10,
                                    save_relative_paths=True)
        # Loading data
        self.data = None
        self.__load_data()

        self.init = None
        # Initializing the model
        self.__init_model()

        # Loading the model checkpoint if exists
        self.__load_model()

        # Summaries
        self.scalar_summary_tags = ['loss', 'acc']
        self.summary_tags = []
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.summary_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)
        self.__init_summaries()

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
            print("First time to train...\n\n")

    ############################################################################################################
    # Summaries methods
    def __init_summaries(self):
        """
        Create the summary part of the graph
        :return:
        """
        with tf.variable_scope('train-summary-per-epoch'):
            for tag in self.scalar_summary_tags:
                self.summary_tags += tag
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

    def __add_summary(self, step, summaries_dict=None, summaries_merged=None):
        """
        Add the summaries to tensorboard
        :param step:
        :param summaries_dict:
        :param summaries_merged:
        :return:
        """
        if summaries_dict is not None:
            summary_list = self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
                                         {self.summary_placeholders[tag]: value for tag, value in
                                          summaries_dict.items()})
            for summary in summary_list:
                self.summary_writer.add_summary(summary, step)
        if summaries_merged is not None:
            self.summary_writer.add_summary(summaries_merged, step)

    ############################################################################################################
    # Train and Test methods
    def train(self):
        #TODO HERE
        train_X, train_Y = self.data.train.next_batch(self.args.batch_size)
        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, self.args.num_epochs + 1, 1):

            # init tqdm and get the epoch value
            tt = tqdm(self.generator(), total=self.num_iterations_training_per_epoch,
                      desc="epoch-" + str(cur_epoch) + "-")

            # init the current iterations
            cur_iteration = 0

            # init acc and loss lists
            loss_list = []
            acc_list = []

            # loop by the number of iterations
            for x_batch, y_batch in tt:

                # get the cur_it for the summary
                cur_it = self.model.global_step_tensor.eval(self.sess)

                # Feed this variables to the network
                feed_dict = {self.model.x_pl: x_batch,
                             self.model.y_pl: y_batch,
                             self.model.is_training: True
                             }

                # Run the feed forward but the last iteration finalize what you want to do
                if cur_iteration < self.num_iterations_training_per_epoch - 1:

                    # run the feed_forward
                    _, loss, acc, summaries_merged = self.sess.run(
                        [self.model.train_op, self.model.loss, self.model.accuracy, self.model.merged_summaries],
                        feed_dict=feed_dict)
                    # log loss and acc
                    loss_list += [loss]
                    acc_list += [acc]
                    # summarize
                    self.add_summary(cur_it, summaries_merged=summaries_merged)

                else:

                    # run the feed_forward
                    _, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                        [self.model.train_op, self.model.loss, self.model.accuracy,
                         self.model.merged_summaries, self.model.segmented_summary],
                        feed_dict=feed_dict)
                    # log loss and acc
                    loss_list += [loss]
                    acc_list += [acc]
                    total_loss = np.mean(loss_list)
                    total_acc = np.mean(acc_list)
                    # summarize
                    summaries_dict = dict()
                    summaries_dict['train-loss-per-epoch'] = total_loss
                    summaries_dict['train-acc-per-epoch'] = total_acc
                    summaries_dict['train_prediction_sample'] = segmented_imgs
                    self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=summaries_merged)

                    # Update the Global step
                    self.model.global_step_assign_op.eval(session=self.sess,
                                                          feed_dict={self.model.global_step_input: cur_it + 1})

                    # Update the Cur Epoch tensor
                    # it is the last thing because if it is interrupted it repeat this
                    self.model.global_epoch_assign_op.eval(session=self.sess,
                                                           feed_dict={self.model.global_epoch_input: cur_epoch + 1})

                    # print in console
                    tt.close()
                    print("epoch-" + str(cur_epoch) + "-" + "loss:" + str(total_loss) + "-" + " acc:" + str(total_acc)[
                                                                                                        :6])

                    # Break the loop to finalize this epoch
                    break

                # Update the Global step
                self.model.global_step_assign_op.eval(session=self.sess,
                                                      feed_dict={self.model.global_step_input: cur_it + 1})

                # update the cur_iteration
                cur_iteration += 1

            # Save the current checkpoint
            if cur_epoch % self.args.save_every == 0:
                self.save_model()

            # Test the model on validation
            if cur_epoch % self.args.test_every == 0:
                self.test_per_epoch(step=self.model.global_step_tensor.eval(self.sess),
                                    epoch=self.model.global_epoch_tensor.eval(self.sess))

        print("Training Finished")

    ############################################################################################################
    # Data methods
    def __load_data(self):
        from tensorflow.examples.tutorials.mnist import input_data
        self.data = input_data.read_data_sets('data/MNIST_data', one_hot=False)
