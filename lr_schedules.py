#holds the lr schedule classes


import tensorflow as tf

class StepChange(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, key_epochs,key_lrs,num_epochs, num_batches):
        #this schedule linearly interpolates between the given points and lrs
        #first param must be e = 0
        super(StepChange, self).__init__()
        self.key_epochs = key_epochs
        self.key_lrs = key_lrs
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.lr = self.key_lrs[0]
        #remove the first element from both lists
        self.key_epochs = self.key_epochs[1:]
        self.key_lrs = self.key_lrs[1:]
        #add the final lr to the end of the list
        self.key_epochs = self.key_epochs + [num_epochs+1]
        self.key_lrs = self.key_lrs + [self.key_lrs[-1]]
        self.index = 0

    @tf.function
    def __call__(self, step):
        index = self.index
        if (step/self.num_batches) >= self.key_epochs[index]:
            self.index += 1
        self.lr = tf.cast(self.key_lrs[index], tf.float64)
        return self.lr


class LinearChange(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,inital_lr, final_lr, num_epochs, num_batches):
        super(LinearChange, self).__init__()
        self.initial_learning_rate = inital_lr
        self.final_learning_rate = final_lr
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.lr = self.initial_learning_rate
    @tf.function
    def __call__(self, step):
        if (step/self.num_batches) >= self.num_epochs:
            self.lr = tf.cast(self.final_learning_rate, tf.float64)
        else:
            self.lr = self.initial_learning_rate + (self.final_learning_rate - self.initial_learning_rate) * ((step/self.num_batches) / self.num_epochs)
        return self.lr

