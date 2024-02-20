from keras import backend as K
from keras.objectives import categorical_crossentropy


if K.image_data_format() == 'channels_last':
	import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4


def rpn_loss_regr(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		if K.image_data_format() == 'channels_first':
			x = y_true[:, 4 * num_anchors:, :, :] - y_pred
			x_abs = K.abs(x)
			x_bool = K.less_equal(x_abs, 1.0)
			return lambda_rpn_regr * K.sum(
				y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
		else:
			x = y_true[:, :, :, 4 * num_anchors:] - y_pred
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

			return lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

	return rpn_loss_regr_fixed_num


def rpn_loss_cls_f1(num_anchors):
	def macro_soft_f1(y_true, y_pred):
		"""Compute the macro soft F1-score as a cost.
		Average (1 - soft-F1) across all labels.
		Use probability values instead of binary predictions.
		
		Args:
			y_true (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
			y_pred (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
			
		Returns:
			cost (scalar Tensor): value of the cost function for the batch
		"""
		
		y_true = tf.cast(y_true, tf.float32)
		y_pred = tf.cast(y_pred, tf.float32)
		tp = tf.reduce_sum(y_pred[:, :, :, :] * y_true[:, :, :, :num_anchors], axis=0)
		fp = tf.reduce_sum(y_pred[:, :, :, :] * (1 - y_true[:, :, :, :num_anchors]), axis=0)
		fn = tf.reduce_sum((1 - y_pred[:, :, :, :]) * y_true[:, :, :, :num_anchors], axis=0)
		soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
		cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
		macro_cost = tf.reduce_mean(cost) # average on all labels
		
		return macro_cost
		
	return macro_soft_f1


def rpn_loss_cls(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		if K.image_data_format() == 'channels_last':
			return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
		else:
			return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, num_anchors:, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

	return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
	return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
	return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
