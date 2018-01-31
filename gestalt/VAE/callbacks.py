import numpy as np
import keras
import keras.callbacks.Callback as Callback

#Just a simple custom callback for terminating on NaN but when there are multiple losses

class TerminateOnNaNMultipleLosses(Callback):
	def _init__(self):
		super(TerminateOnNaNMultipleLosses,self).__init__()


	def on_batch_end(self, batch, logs=None):
		logs = logs or []
		losses = logs.get('loss')
		NaNDetected = False
		if losses is not None:
			for loss in losses:
				if np.isnan(loss) or np.isinf(loss):
					print('Batch %d: Invalid Loss %d, terminating training' %(batch) %(loss))
					NaNDetected = True

		if NaNDetected:
			self.model.stop_training = true
