import tensorflow as tf


class Model:
	def __init__(self,
				 hidden_num,
				 inputs):
	"""
	inputs: (batch_num x elem_num)
	"""

		self.batch_num = inputs[]
		self.enc_cell = tf.nn.rnn_cell.LSTMCell(hidden_num)
		self.dec_cell = 

		time_steps = 50  #fixed, should use dynamic rnn
		batch_size = 50
		num_features = 5  #5 roles, more features to be added
		lstm_size = 32 #enough to encode sequence

		words_in_dataset = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])
		lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
		# Initial state of the LSTM memory.
		hidden_state = tf.zeros([batch_size, lstm.state_size])
		current_state = tf.zeros([batch_size, lstm.state_size])
		state = hidden_state, current_state
		probabilities = []
		loss = 0.0
		for current_batch_of_words in words_in_dataset:
		    # The value of state is updated after processing each batch of words.
		    output, state = lstm(current_batch_of_words, state)
		    #inputs: 2-D tensor with shape [batch_size x input_size].
		    #state: if self.state_size is an integer, this should be a 2-D Tensor with shape [batch_size x self.state_size]. 
		    #		Otherwise, if self.state_size is a tuple of integers, this should be a tuple with shapes [batch_size x s] for s in self.state_size.
		    

		    # The LSTM output can be used to make next word predictions
		    logits = tf.matmul(output, softmax_w) + softmax_b
		    probabilities.append(tf.nn.softmax(logits))
		    loss += loss_function(probabilities, target_words)


		# Embedding
		embedding_encoder = variable_scope.get_variable("embedding_encoder", 
										[src_vocab_size, embedding_size], ...)
		# Look up embedding:
		#   encoder_inputs: [max_time, batch_size]
		#   encoder_emb_inp: [max_time, batch_size, embedding_size]
		encoder_emb_inp = embedding_ops.embedding_lookup(embedding_encoder, 
														 encoder_inputs)

		encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
		# Run Dynamic RNN
		#   encoder_outputs: [max_time, batch_size, num_units]
		#   encoder_state: [batch_size, num_units]
		encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
										encoder_cell, encoder_emb_inp,
		    							sequence_length=source_sequence_length, 
		    							time_major=True)
		# Build RNN cell
		decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
		# Helper
		helper = tf.contrib.seq2seq.TrainingHelper(
		    				decoder_emb_inp, decoder_lengths, time_major=True)
		# Decoder
		decoder = tf.contrib.seq2seq.BasicDecoder(
									    decoder_cell, helper, encoder_state,
									    output_layer=projection_layer)
		# Dynamic decoding
		outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
		logits = outputs.rnn_output

		