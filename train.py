import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
# These are the local imports. We import that from our directory 
# driving_data is for reading our dataset
import driving_data
# model is out tensorflow model. check the model graph here. https://imgur.com/IuBJdKe
import model
# the path for our trained model. In case there is a trained model already we will import that and start training with that. If you want to you can also start from scratch.
LOGDIR = './save'
# Tensorflow Session. Read more here https://www.tensorflow.org/api_docs/python/tf/Session
sess = tf.InteractiveSession()
# This is our normalization function. We use L2 and now we define a constant for that. 
L2NormConst = 0.001

train_vars = tf.trainable_variables()

loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.initialize_all_variables())

# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op =  tf.summary.merge_all()

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
# 30 epochs is a good number. You can in or decrease baed on your computation power.
epochs = 30
batch_size = 100

# Training with those values takes about an hour on a GTX 1070 with CUDA 10.0
for epoch in range(epochs):
  for i in range(int(driving_data.num_images/batch_size)):
    xs, ys = driving_data.LoadTrainBatch(batch_size)
    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
    if i % 10 == 0:
      xs, ys = driving_data.LoadValBatch(batch_size)
      # The Step value is calculated by: epoch * batchsize + i
      loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # write logs at every iteration. This is to make sure that nothing goes wrong 
    summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)

    if i % batch_size == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
print("You are done now. You can execute run_dataset.py to see a live demo or run.py to use a webcam with live video feed, ideally mounted in your car.")
