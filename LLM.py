

#File to hold model class

import tensorflow as tf
from tensorflow.keras import layers,losses



class Model():
    def __init__(config,train_ds):
        self.max_tokens = config.vocab_size
        self.sequence_length = config.max_length
        self.model = None
        self.loss = None
        self.optimizer = None
        self.train_loss = None
        self.train_accuracy = None
        self.test_loss = None
        self.test_accuracy = None
        self.build_model(train_ds)

    def __custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')


    def __vectorize_text(self, text, label):
            text = tf.expand_dims(text, -1)
            return self.vectorize_layer(text), label

    def build_model(self, train_ds):
        

        self.vectorize_layer = layers.TextVectorization(
            standardize=self.__custom_standardization,
            max_tokens=self.max_tokens,
            output_mode='int',
            output_sequence_length=self.sequence_length)
        
        # Make a text-only dataset (no labels) and call adapt to build the vocabulary.
        text_ds = train_ds.map(lambda x, y: x)
        self.vectorize_layer.adapt(text_ds)

        # Create the model#ADD MRE
        


