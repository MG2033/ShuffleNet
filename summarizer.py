import tensorflow as tf


class Summarizer:
    """The class responsible for Tensorboard summaries such as loss, and classification accuracy"""

    def __init__(self, sess, summary_dir):
        # Summaries
        self.sess = sess
        self.scalar_summary_tags = ['loss', 'acc', 'test-loss', 'test-acc']
        self.summary_tags = []
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        self.__init_summaries()

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

    def add_summary(self, step, summaries_dict=None, summaries_merged=None):
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
