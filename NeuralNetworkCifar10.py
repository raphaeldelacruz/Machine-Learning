import tensorflow as tf
tf.executing_eagerly()
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt


# Hyperparameters
training_epochs = 20            
learning_rate = 0.001
train_batch_size = 5000
test_batch_size = 1000

img_h = img_w = 32              # cifar10 images are 32
n_input = 784                   # 32x32 = 1024
n_hidden_1 = 64
n_hidden_2 = 32
n_classes = 10

cifar10 = tf.keras.datasets.cifar10


##########################
### Neural Network DEFINITION
##########################

cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(train_batch_size).batch(train_batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(test_batch_size).batch(test_batch_size)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(n_hidden_1, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(n_hidden_2, activation='relu')
    self.d2 = Dense(n_classes)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

cost_vector = []
train_acc_vector = []
test_acc_vector = []

for epoch in range(training_epochs):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )
  
  cost_vector.append(train_loss.result())
  train_acc_vector.append(1-train_accuracy.result())
  test_acc_vector.append(1-test_accuracy.result())
  
##########################
### Visualizing The Result
##########################

Fontsize = 12
fig, _axs = plt.subplots(nrows=1, ncols=2)
axs = _axs.flatten()

l01, = axs[0].plot(range(training_epochs), cost_vector,'g')
axs[0].set_xlabel('Epoch',fontsize=Fontsize)
axs[0].set_ylabel('Cost',fontsize=Fontsize)
axs[0].grid(True)

l11, = axs[1].plot(range(training_epochs), train_acc_vector,'b')
l12, = axs[1].plot(range(training_epochs), test_acc_vector,'r')
axs[1].set_xlabel('Epoch',fontsize=Fontsize)
axs[1].set_ylabel('Error',fontsize=Fontsize)
axs[1].grid(True)
axs[1].legend(handles = [l11, l12], labels = ['train error', 'test error'],loc = 'upper right', fontsize=Fontsize)

plt.show()