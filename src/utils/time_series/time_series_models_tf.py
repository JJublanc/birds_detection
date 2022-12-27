import tensorflow as tf


class Baseline(tf.keras.Model):
	def __init__(self, label_index=None):
		super().__init__()
		self.label_index = label_index

	def call(self, inputs):
		if self.label_index is None:
			return inputs
		result = inputs[:, :, self.label_index]
		return result[:, :, tf.newaxis]


def get_dense_model(patience=2):
	dense_model = tf.keras.Sequential(
		[
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(units=1,),
		]
	)

	early_stopping = tf.keras.callbacks.EarlyStopping(
		monitor="val_loss", patience=patience, mode="min"
	)

	dense_model.compile(
		loss="mse",
		optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9),
		metrics=[tf.keras.metrics.MeanAbsoluteError()],
	)

	return dense_model, early_stopping


def get_lstm(patience=2):
	model = tf.keras.models.Sequential([
		# Shape [batch, time, features] => [batch, time, lstm_units]
		tf.keras.layers.LSTM(32, return_sequences=True),
		# Shape => [batch, time, features]
		tf.keras.layers.Dense(units=32, activation="relu"),
		tf.keras.layers.Dropout(0.4),
		tf.keras.layers.Dense(units=1)
	])

	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
	                                                  patience=patience,
	                                                  mode='min')

	model.compile(loss=tf.keras.losses.MeanSquaredError(),
	              optimizer=tf.keras.optimizers.RMSprop(),
	              metrics=[tf.keras.metrics.MeanAbsoluteError()])

	return model, early_stopping


def get_lstm2(patience=2):
	model = tf.keras.models.Sequential([
	  tf.keras.layers.LSTM(64, return_sequences=True),
	  tf.keras.layers.LSTM(64, return_sequences=True),
	  tf.keras.layers.Dense(30, activation="relu"),
	  tf.keras.layers.Dense(10, activation="relu"),
	  tf.keras.layers.Dense(1),
	])

	lr_schedule = tf.keras.callbacks.LearningRateScheduler(
	    lambda epoch: 1e-8 * 10**(epoch / 20))

	optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)
	model.compile(loss=tf.keras.losses.Huber(),
	              optimizer=optimizer,
	              metrics=["mae"])

	return model, lr_schedule
