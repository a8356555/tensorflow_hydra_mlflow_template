import tensorflow as tf
import sys

class AccumGradModel(tf.keras.Model):
    def __init__(self, accum_iters=1, total_batches=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accum_iters = tf.constant(accum_iters, dtype=tf.int32)
        self.total_batches = tf.constant(total_batches, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.step_in_epoch = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]
    
    def train_step(self, data):
        self.n_acum_step.assign_add(1)
        self.step_in_epoch.assign_add(1)
        # tf.print('\naccum step: ', self.n_acum_step, self.n_acum_step%self.accum_iters, self.accum_iters, output_stream=sys.stderr)
        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            # loss = loss / tf.cast(self.accum_iters, tf.float32)
        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the accum_iters then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step%self.accum_iters, 0), 
            self.apply_accu_gradients, 
            lambda: None
        )
        tf.cond(tf.equal(self.step_in_epoch%self.total_batches, 0), 
            self.apply_epoch_end_gradients, 
            lambda: None
        )

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        tf.print('apply accumulated gradients', output_stream=sys.stderr)
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))
            
            
    def apply_epoch_end_gradients(self):
        # apply accumulated gradients
        tf.print('apply accumulated gradients epoch end', output_stream=sys.stderr)
        self.apply_accu_gradients()
        self.step_in_epoch.assign(0)