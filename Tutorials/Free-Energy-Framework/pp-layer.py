# so this is just meant to be a very simple and testy interpretation ofthe pp layer
# mostly to get my own head around this. Not for any kind of serious implementatoin yet
# we can work on that in future days, but just to see if we understand it

# let's think aboutstructure here.... the layer has two outputs which must be available to hte the layers. It propagates prediction errors upwards, and predictions downwards. That's what it needs to do. So what we will do
# wait a second. I'm kind of confused by this formula to be honest, since to produce prediction errors we need predictions, and to produce predictions we need prediction errors. and that's just dagnabbit. NOpe, we don't actually. that's good, I think. We just need oru old predictions. Okay, that's good and nice. phew. We got lucky with that one! I think. I'm not totally sure of the order of operations
# so what we do is and we have some kind of activation functoin, which must be differentiable, but other than that isn't that difficult to be honest. Or should wealways have things through the derivative of the activation function. whence o we push things through? idk really I must admit. I think we only use the weights andstuff we have available, and I'm not sure. we also need to figure out how to compute the precision inverse locally as well, but that can come and wait for a bit!
# I can't believe I'ev spent the whole day on this and have basically nothingto show for it... dagnabbit. The day has gone by so fast. I kind of understand how it works now, and have built the model, but we definitely don't have anything we can apply to autism or whatever for another few days... dagnabbit. and less working time also, I feel!

# I'm really not sure how to build a pp framework. But that's a software challenge for AFTEr I get the autism paper and stuff done, I realy do not know dagnabbit. but I hope it will be good?

import numpy as np

class PPLayer(object):
	def __init__(self, input_pred_errors, input_predictions):
		# what params do we have here that I need?
		#let's do some sensible kind of iteration, idk
		self.input_pred_errors = input_pred_errors
		self.input_predictions = input_predictions
		self.pred_errs
		self.predictions
		self.activation_function
		self.activation_function_derivative
		self.weights
		self.precision
		self.lrate


	#first I think we need to compute our predictions
	#this is basically the forwad step, I think
	def compute_predictions(self, input_pred_errs):
			delta_pred = np.dot(activation_function_derivative(self.predictions), input_pred_errs) - self.pred_errs
			new_pred= self.predictions + self.lrate * self.delta_pred
			self.predictions = new_pred
			return new_pred

	def compute_prediction_errors(self, input_predictions):
		delta_pred_err = self.predictions - np.dot(self.weights, input_predictions) - np.dot(self.precision, self.pred_errs)
		new_pred_err = self.pred_errs + (self.lrate * self.delta_pred_err)
		self.pred_errs = new_pred_err
		return new_pred_err

	def pass_down_predictions(self):
		return self.activation_function(self.predictions)

	def pass_up_prediction_errors(self):
		return np.dot(self.weights.T, self.pred_errs)


		
