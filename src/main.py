import os
import numpy as np
import random
random_seed = 5  #random.randint(0, 10000)
print("Selected random seed is : ", random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
import keras
import pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from functools import partial
import sys
sys.path.insert(0, './logic/lib/')
sys.path.insert(0, './logic/python/')
from learn_add import *
import LogicLayer as LL
from keras import optimizers
from models import NN_model

import argparse

from map_generator import map_generator

DEBUG = True


def get_img_data(src_path, labels, shape=(28, 28, 1)):
	print("\n** Now getting all images **")
	#image
	X = []
	#label
	Y = []
	h = shape[0]
	w = shape[1]
	d = shape[2]
	#index = [0,1,2,3]
	for (index, label) in enumerate(labels):
		label_folder_path = os.path.join(src_path, label)
		for p in os.listdir(label_folder_path):
			image_path = os.path.join(label_folder_path, p)
			if d == 1:
				mode = 'I'
			else:
				mode = 'RGB'
			image = Image.open(image_path).convert(mode).resize((h, w))
			X.append((np.array(image) - 127) * (1 / 128.))
			Y.append(index)

	X = np.array(X)
	Y = np.array(Y)

	index = np.array(list(range(len(X))))
	np.random.shuffle(index)
	X = X[index]
	Y = Y[index]

	assert (len(X) == len(Y))
	print("Total data size is :", len(X))
	# normalize
	X = X.reshape(-1, h, w, d)
	Y = np_utils.to_categorical(Y, num_classes=len(labels))
	return X, Y


def get_nlm_net(labels_num, shape=(28, 28, 1), model_name="LeNet5"):
	assert model_name == "LeNet5"
	#if model_name == "LeNet5":
	#    return NN_model.get_LeNet5_net(labels_num, shape)
	d = shape[2]
	if d == 1:
		return NN_model.get_LeNet5_net(labels_num, shape)
	else:
		return NN_model.get_cifar10_net(labels_num, shape)


def net_model_test(src_path, labels, src_data_name, shape=(28, 28, 1)):
	print("\n** Now use the model to train and test the images **")

	file_name = '%s_correct_model_weights.hdf5' % src_data_name
	if os.path.exists(file_name):
		print("Model file exists, skip model testing step.")
		return
	d = shape[2]
	X, Y = get_img_data(src_path, labels, shape)

	# train:test = 5:1
	X_train = X[:len(X) // 5 * 1]
	y_train = Y[:len(Y) // 5 * 1]
	X_test = X[len(X) // 5 * 1:]
	y_test = Y[len(Y) // 5 * 1:]

	print("Train data size:", len(X_train))
	print("Test data size:", len(X_test))

	model = get_nlm_net(len(labels), shape)
	#Add decay
	opt_rms = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
	model.compile(optimizer=opt_rms,
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])
	print('Training...')

	if d == 1:
		model.fit(X_train,
				  y_train,
				  epochs=30,
				  batch_size=32,
				  validation_data=(X_test, y_test))
	else:
		#data augmentation
		datagen = ImageDataGenerator(
			rotation_range=15,
			width_shift_range=0.1,
			height_shift_range=0.1,
			horizontal_flip=True,
		)
		datagen.fit(X_train)
		model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),\
		  steps_per_epoch=X_train.shape[0]//32,epochs=100,verbose=1,validation_data=(X_test,y_test))
	model.save_weights(file_name)
	print("Model saved to ", file_name)

	print('\nTesting...')
	loss, accuracy = model.evaluate(X_test, y_test)
	print('Test loss: ', loss)
	print('Test accuracy: ', accuracy)


def net_model_pretrain(src_path, labels, src_data_name, shape=(28, 28, 1)):
	print("\n** Now use autoencoder to pretrain the model **")

	file_name = '%s_pretrain_weights.hdf5' % src_data_name
	h = shape[0]
	w = shape[1]
	d = shape[2]
	if os.path.exists(file_name):
		print("Pretrain file exists, skip pretrain step.")
		return

	X, _ = get_img_data(src_path, labels, shape)
	print("There are %d pretrain images" % len(X))

	if d == 1:
		Y = X.copy().reshape(-1, h * w * d)
		model = NN_model.get_LeNet5_autoencoder_net(len(labels), shape)
	else:
		Y = X
		model = NN_model.get_autoencoder_net(len(labels), input_shape=shape)
	model.compile(optimizer='rmsprop',
				  loss='mean_squared_error',
				  metrics=['accuracy'])

	print('Pretraining...')
	model.fit(X, Y, epochs=10, batch_size=64)
	model.save_weights(file_name)
	print("Model saved to ", file_name)


def LL_init(pl_file_path):
	print("\n** Initializing prolog **")
	assert os.path.exists(pl_file_path), "%s is not exist" % pl_file_path
	# must initialise prolog engine first!
	LL.init("-G10g -M6g")
	# consult the background knowledge file
	LL.consult(pl_file_path)
	#test if stack changed
	LL.call("prolog_stack_property(global, limit(X)), writeln(X)")
	print("Prolog has alreadly initialized")


def divide_equation_by_len(equations):
	'''
	Divide equations by length
	equations has alreadly been sorted, so just divide it by equation's length
	'''
	equations_by_len = list()
	start = 0
	for i in range(1, len(equations) + 1):
		#print(len(equations[i]))
		if i == len(equations) or len(equations[i]) != len(equations[start]):
			equations_by_len.append(equations[start:i])
			start = i
	return equations_by_len


def split_equation(equations_by_len, prop_train, prop_val):
	'''
	Split the equations in each length to training and validation data according to the proportion
	'''
	train = []
	val = []
	all_prop = prop_train + prop_val
	for equations in equations_by_len:
		random.shuffle(equations)
		train.append(equations[:len(equations) // all_prop * prop_train])
		val.append(equations[len(equations) // all_prop * prop_train:])
		#print(len(equations[:len(equations)//all_prop*prop_train]))
		#print(len(equations[len(equations)//all_prop*prop_train:]))
	return train, val


def constraint(solution, min_var, max_var):
	'''
	Constrain how many position to abduce
	'''
	x = solution.get_x()
	#print(x)
	return (max_var - x.sum()) * (x.sum() - min_var)


def get_abduced_result(exs, maps, no_change):
	consist_re = None
	consist_res_max = (-1, [])
	exs_shape = [len(tmpex) for tmpex in exs]
	# Constrain zoopt modify at least 1 bit and at most 10 bits
	c = partial(constraint, min_var=1, max_var=10)
	#Try all mappings
	for m in maps:
		# Check if it can abduce rules without changing any labels
		if no_change:
			#Assuming that each equation is the same length
			consist_res = consistent_score_sets(exs, [0] * sum(exs_shape), m)
		# Find the possible wrong position in symbols and Abduce the right symbol through logic module
		else:
			# Use zoopt to optimize
			sol = opt_var_ids_sets_constraint(exs, m, c)
			# Get the result
			consist_res = consistent_score_sets(exs,
												[int(i) for i in sol.get_x()],
												m)
		if consist_res[0] > consist_res_max[0]:
			consist_res_max = consist_res

	# consistent_score_sets() returns consist_res=(score, eq_sets). the score is the function value for zoopt, so we only care about the second element when printting result
	max_consist_num = 0
	#Choose the result that has max num of consistent examples
	for re in consist_res_max[1]:
		if len(re.consistent_ex_ids) > max_consist_num:
			max_consist_num = len(re.consistent_ex_ids)
			consist_re = re

	if no_change:
		if max_consist_num == len(exs):
			#print("#It can abduce rules without changing any labels")
			return consist_re
		return None
	else:
		return consist_re


def get_equations_labels(model,
						 equations,
						 labels,
						 abduced_map=None,
						 no_change=False,
						 shape=(28, 28, 1)):
	'''
	Get the model's abduced output through abduction
	model: NN model
	equations: equation images
	labels: [0,1,10,11] now  only use len(labels)
	maps: abduced map like [0:'+',1:'=',2:0,3:1] if None, then try all possible mappings
	no_change: if True, it indicates that do not abduce, only get rules from equations
	shape: shape of image
	'''
	h = shape[0]
	w = shape[1]
	d = shape[2]
	exs = []

	for e in equations:
		exs.append(
			np.argmax(model.predict(e.reshape(-1, h, w, d)), axis=1).tolist())
	if no_change == False:
		print("\n\nThis is the model's current label:")
		print(exs)
	if abduced_map is None:
		maps = gen_mappings(
			[0, 1, 2, 3],
			['+', '=', 0, 1
			 ])  # All possible mappings from label to signs(0 1 + =)
		#maps = list(map_generator(['+', '='], 2)) # only consider '+' and '=' mapping
	else:
		maps = [abduced_map]

	# Check if it can abduce rules without changing any labels
	consist_re = get_abduced_result(exs, maps, True)
	if consist_re is None:
		if no_change == True:
			return (None, None)
	else:
		return (consist_re, consist_re.to_feature().rules.py())

	# Find the possible wrong position in symbols and Abduce the right symbol through logic module
	consist_re = get_abduced_result(exs, maps, False)
	if consist_re is None:
		return (None, None)

	feat = consist_re.to_feature(
	)  # Convert consistent result to add rules my_op, will be used to train the decision MLP
	rule_set = feat.rules.py()

	if DEBUG:
		print('****Consistent instance:')
		print('consistent examples:', end='\t')
		# Max consistent subset's index in original exs list
		print(consist_re.consistent_ex_ids)
		print('mapping:', end='\t')
		# Mapping used in abduction
		print(consist_re.abduced_map)
		print('abduced examples:', end='\t')
		# Modified label sequence after abduction, will be used to retrain CNN
		print(consist_re.abduced_exs)
		print('abduced examples(after mapping):', end='\t')
		# abduced_exs after using mapping
		print(consist_re.abduced_exs_mapped)

		print('****Learned feature:')
		print('rules: ', end='\t')
		print(rule_set)

	return (consist_re, rule_set)


def get_mlp_vector(equation, model, rules, abduced_map, shape=(28, 28, 1)):
	h = shape[0]
	w = shape[1]
	d = shape[2]
	model_output = np.argmax(model.predict(equation.reshape(-1, h, w, d)),
							 axis=1)
	model_labels = []
	for out in model_output:
		model_labels.append(abduced_map[out])
	#print(model_labels)
	vector = []
	for rule in rules:
		ex = LL.PlTerm(model_labels)
		f = LL.PlTerm(rule)
		if LL.evalInstFeature(ex, f):
			vector.append(1)
		else:
			vector.append(0)
	return vector


def get_mlp_data(equations_true,
				 equations_false,
				 base_model,
				 out_rules,
				 abduced_map,
				 shape=(28, 28, 1)):
	mlp_vectors = []
	mlp_labels = []
	for equation in equations_true:
		mlp_vectors.append(
			get_mlp_vector(equation, base_model, out_rules, abduced_map,
						   shape))
		mlp_labels.append(1)
	for equation in equations_false:
		mlp_vectors.append(
			get_mlp_vector(equation, base_model, out_rules, abduced_map,
						   shape))
		mlp_labels.append(0)
	mlp_vectors = np.array(mlp_vectors)
	mlp_labels = np.array(mlp_labels)
	return mlp_vectors, mlp_labels


def get_file_data(src_data_file, prop_train, prop_val):
	with open(src_data_file, 'rb') as f:
		equations = pickle.load(f)
	input_file_true = equations['train:positive']
	input_file_false = equations['train:negative']
	input_file_true_test = equations['test:positive']
	input_file_false_test = equations['test:negative']

	equations_true_by_len = divide_equation_by_len(input_file_true)
	equations_false_by_len = divide_equation_by_len(input_file_false)
	equations_true_by_len_test = divide_equation_by_len(input_file_true_test)
	equations_false_by_len_test = divide_equation_by_len(input_file_false_test)
	#train:validation:test = prop_train:prop_val
	equations_true_by_len_train, equations_true_by_len_validation = split_equation(
		equations_true_by_len, prop_train, prop_val)
	equations_false_by_len_train, equations_false_by_len_validation = split_equation(
		equations_false_by_len, prop_train, prop_val)

	for equations_true in equations_true_by_len:
		print(
			"There are %d true training and validation equations of length %d"
			% (len(equations_true), len(equations_true[0])))
	for equations_false in equations_false_by_len:
		print(
			"There are %d false training and validation equations of length %d"
			% (len(equations_false), len(equations_false[0])))
	for equations_true in equations_true_by_len_test:
		print("There are %d true testing equations of length %d" %
			  (len(equations_true), len(equations_true[0])))
	for equations_false in equations_false_by_len_test:
		print("There are %d false testing equations of length %d" %
			  (len(equations_false), len(equations_false[0])))

	return (equations_true_by_len_train, equations_true_by_len_validation,
			equations_false_by_len_train, equations_false_by_len_validation,
			equations_true_by_len_test, equations_false_by_len_test)


def get_percentage_precision(base_model, select_equations, consist_re, shape):
	h = shape[0]
	w = shape[1]
	d = shape[2]
	consistent_ex_ids = consist_re.consistent_ex_ids
	abduced_map = consist_re.abduced_map

	if DEBUG:
		print("Abduced labels:")
		for ex in consist_re.abduced_exs_mapped:
			for c in ex:
				print(c, end='')
			print(' ', end='')
		print("\nCurrent model's output:")

	model_labels = []
	for e in select_equations[consistent_ex_ids]:
		hat_y = np.argmax(base_model.predict(e.reshape(-1, h, w, d)), axis=1)
		model_labels.append(hat_y)
		info = ""
		for y in hat_y:
			info += str(abduced_map[y])
		if DEBUG:
			print(info, end=' ')
	model_labels = np.concatenate(np.array(model_labels)).flatten()

	abduced_labels = flatten(consist_re.abduced_exs)
	batch_label_model_precision = (model_labels == abduced_labels).sum() / (
		len(model_labels))
	consistent_percentage = len(consistent_ex_ids) / len(select_equations)
	print("\nBatch label model precision:", batch_label_model_precision)
	print("Consistent percentage:", consistent_percentage)

	return consistent_percentage, batch_label_model_precision


def get_rules_from_data(base_model, equations_true, labels, abduced_map, shape,
						LOGIC_OUTPUT_DIM, SAMPLES_PER_RULE):
	out_rules = []
	for i in range(LOGIC_OUTPUT_DIM):
		while True:
			select_index = np.random.randint(len(equations_true),
											 size=SAMPLES_PER_RULE)
			select_equations = np.array(equations_true)[select_index]
			_, rule = get_equations_labels(base_model, select_equations,
										   labels, abduced_map, True, shape)
			if rule != None:
				break
		out_rules.append(rule)
	return out_rules


def abduce_and_train(base_model, equations_true, labels, abduced_map, shape,
					 SELECT_NUM, BATCHSIZE, NN_EPOCHS):
	h = shape[0]
	w = shape[1]
	d = shape[2]
	#Randomly select several equations
	select_index = np.random.randint(len(equations_true), size=SELECT_NUM)
	select_equations = np.array(equations_true)[select_index]

	consist_re, _ = get_equations_labels(base_model, select_equations, labels,
										 abduced_map, False, shape)
	# Can not abduce
	if consist_re is None:
		return (0, 0, abduced_map)
	consistent_ex_ids = consist_re.consistent_ex_ids
	equations_labels = consist_re.abduced_exs
	abduced_map = consist_re.abduced_map

	train_pool_X = np.concatenate(select_equations[consistent_ex_ids]).reshape(
		-1, h, w, d)
	train_pool_Y = np_utils.to_categorical(
		flatten(equations_labels),
		num_classes=len(labels))  # Convert the symbol to network output
	assert (len(train_pool_X) == len(train_pool_Y))
	print("\nTrain pool size is :", len(train_pool_X))
	print("Training...")
	#cifar10  data augmentation
	if d > 1:
		datagen = ImageDataGenerator(
			rotation_range=15,
			width_shift_range=0.1,
			height_shift_range=0.1,
			horizontal_flip=True,
		)
		datagen.fit(train_pool_X)
		base_model.fit_generator(datagen.flow(
			train_pool_X, train_pool_Y, batch_size=train_pool_X.shape[0]),
								 steps_per_epoch=1,
								 epochs=NN_EPOCHS)
	else:
		base_model.fit(train_pool_X,
					   train_pool_Y,
					   batch_size=BATCHSIZE,
					   epochs=NN_EPOCHS,
					   verbose=0)

	consistent_percentage, batch_label_model_precision = get_percentage_precision(
		base_model, select_equations, consist_re, shape)
	return consistent_percentage, batch_label_model_precision, abduced_map


def validation(base_model, equations_true, equations_false, equations_true_val,
			   equations_false_val, labels, abduced_map, shape,
			   LOGIC_OUTPUT_DIM, SAMPLES_PER_RULE, MLP_BATCHSIZE, MLP_EPOCHS):
	#Generate several rules
	#Get training data and label
	#Train mlp
	#Evaluate mlp
	print("Now checking if we can go to next course")
	out_rules = get_rules_from_data(base_model, equations_true, labels,
									abduced_map, shape, LOGIC_OUTPUT_DIM,
									SAMPLES_PER_RULE)
	#print(out_rules)

	#Prepare MLP training data
	mlp_train_vectors, mlp_train_labels = get_mlp_data(equations_true,
													   equations_false,
													   base_model, out_rules,
													   abduced_map, shape)
	index = np.array(list(range(len(mlp_train_labels))))
	np.random.shuffle(index)
	mlp_train_vectors = mlp_train_vectors[index]
	mlp_train_labels = mlp_train_labels[index]

	best_accuracy = 0
	#Try three times to find the best mlp
	for i in range(3):
		#Train MLP
		print("Training mlp...")
		mlp_model = NN_model.get_mlp_net(LOGIC_OUTPUT_DIM)
		mlp_model.compile(loss='binary_crossentropy',
						  optimizer='rmsprop',
						  metrics=['accuracy'])
		mlp_model.fit(mlp_train_vectors,
					  mlp_train_labels,
					  epochs=MLP_EPOCHS,
					  batch_size=MLP_BATCHSIZE,
					  verbose=0)
		#Prepare MLP validation data
		mlp_val_vectors, mlp_val_labels = get_mlp_data(equations_true_val,
													   equations_false_val,
													   base_model, out_rules,
													   abduced_map, shape)
		#Get MLP validation result
		result = mlp_model.evaluate(mlp_val_vectors,
									mlp_val_labels,
									batch_size=MLP_BATCHSIZE,
									verbose=0)
		print("MLP validation result:", result)
		accuracy = result[1]

		if accuracy > best_accuracy:
			best_accuracy = accuracy
	return best_accuracy


def get_all_mlp_data(equations_true_by_len_train, equations_false_by_len_train,
					 base_model, out_rules, abduced_map, shape,
					 EQUATION_LEN_CNT):
	mlp_train_vectors = []
	mlp_train_labels = []
	#for each length of test equations
	for equations_type in range(EQUATION_LEN_CNT):
		mlp_train_len_vectors, mlp_train_len_labels = get_mlp_data(
			equations_true_by_len_train[equations_type],
			equations_false_by_len_train[equations_type], base_model,
			out_rules, abduced_map, shape)
		if equations_type == 0:
			mlp_train_vectors = mlp_train_len_vectors.copy()
			mlp_train_labels = mlp_train_len_labels.copy()
		else:
			mlp_train_vectors = np.concatenate(
				(mlp_train_vectors, mlp_train_len_vectors), axis=0)
			mlp_train_labels = np.concatenate(
				(mlp_train_labels, mlp_train_len_labels), axis=0)

	index = np.array(list(range(len(mlp_train_labels))))
	np.random.shuffle(index)
	mlp_train_vectors = mlp_train_vectors[index]
	mlp_train_labels = mlp_train_labels[index]
	return mlp_train_vectors, mlp_train_labels


def test_nn_model(model, src_path, labels, input_shape):
	print("\nNow test the NN model")
	'''
	model = NN_model.get_LeNet5_net(len(labels))
	model.load_weights('mnist_images_nlm_weights_3.hdf5')
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	'''
	best_accuracy = 0
	X, Y = get_img_data(src_path, labels, input_shape)
	maps = gen_mappings([0, 1, 2, 3], [0, 1, 2, 3])
	print('\nTesting...')
	# We don't know the map, so we try all maps and get the best accuracy
	for mapping in maps:
		real_Y = []
		for y in Y:
			real_Y.append(mapping[np.argmax(y)])
		Y_cate = np_utils.to_categorical(real_Y, num_classes=len(labels))
		loss, accuracy = model.evaluate(X, Y_cate, verbose=0)
		if accuracy > best_accuracy:
			best_accuracy = accuracy
	print('Neural network perception accuracy: ', best_accuracy)


def nlm_main_func(labels, src_data_name, src_data_file, pl_file_path, shape,
				  args):
	LL_init(pl_file_path)

	h = shape[0]
	w = shape[1]
	d = shape[2]
	abduced_map = None

	#LOGIC_OUTPUT_DIM = 50 #The mlp vector has 50 dimensions
	LOGIC_OUTPUT_DIM = args.LOGIC_OUTPUT_DIM
	#EQUATION_MAX_LEN = 8 #Only learn the equations of length 5-8
	EQUATION_MAX_LEN = args.EQUATION_MAX_LEN
	EQUATION_LEN_CNT = EQUATION_MAX_LEN - 4  #equations index is 0-4
	#SELECT_NUM = 10 #Select 10 equations to abduce rules
	SELECT_NUM = args.SELECT_NUM
	#
	## Proportion of train and validation = 3:1
	#PROP_TRAIN = 3
	PROP_TRAIN = args.PROP_TRAIN
	#PROP_VALIDATION = 1
	PROP_VALIDATION = args.PROP_VALIDATION
	#
	#CONSISTENT_PERCENTAGE_THRESHOLD = 0.9
	CONSISTENT_PERCENTAGE_THRESHOLD = args.CONSISTENT_PERCENTAGE_THRESHOLD
	#BATCH_LABEL_MODEL_PRECISION_THRESHOLD = 0.9 #If consistent percentage is higher than 0.9 and model precision higher than 0.9, then the condition is satisfied
	BATCH_LABEL_MODEL_PRECISION_THRESHOLD = args.BATCH_LABEL_MODEL_PRECISION_THRESHOLD
	#CONDITION_CNT_THRESHOLD = 5       #If the condition has been satisfied 5 times, the start validation
	CONDITION_CNT_THRESHOLD = args.CONDITION_CNT_THRESHOLD
	#NEXT_COURSE_ACC_THRESHOLD = 0.86  #If the validation accuracy of a course higher than the threshold, then go to next course
	NEXT_COURSE_ACC_THRESHOLD = args.NEXT_COURSE_ACC_THRESHOLD

	#SAMPLES_PER_RULE = 3 # Use 3 samples to abduce a rule when training mlp
	SAMPLES_PER_RULE = args.SAMPLES_PER_RULE
	#NN_BATCHSIZE = 32    # Batch size of neural network
	NN_BATCHSIZE = args.NN_BATCHSIZE
	#NN_EPOCHS = 10       # Epochs of neural network
	NN_EPOCHS = args.NN_EPOCHS
	#MLP_BATCHSIZE = 128  # Batch size of mlp
	MLP_BATCHSIZE = args.MLP_BATCHSIZE
	#MLP_EPOCHS = 60      # Epochs of mlp
	MLP_EPOCHS = args.MLP_EPOCHS

	# Get NN model and compile
	if d == 1:
		t_model = NN_model.get_LeNet5_autoencoder_net(len(labels), shape)
	else:
		t_model = NN_model.get_cifar10_net(len(labels), shape)
	t_model.load_weights('%s_pretrain_weights.hdf5' % src_data_name)
	base_model = get_nlm_net(len(labels), shape)
	for i in range(len(base_model.layers)):
		base_model.layers[i].set_weights(t_model.layers[i].get_weights())
	opt_rms = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
	base_model.compile(optimizer=opt_rms,
					   loss='categorical_crossentropy',
					   metrics=['accuracy'])

	# Get file data
	equations_true_by_len_train,equations_true_by_len_validation,equations_false_by_len_train,\
	equations_false_by_len_validation,equations_true_by_len_test,equations_false_by_len_test = get_file_data(src_data_file, PROP_TRAIN, PROP_VALIDATION)

	# Start training / for each length of equations
	for equations_type in range(EQUATION_LEN_CNT - 1):
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		print("LENGTH: ", 5 + equations_type, " to ", 5 + equations_type + 1)
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		equations_true = equations_true_by_len_train[equations_type]
		equations_false = equations_false_by_len_train[equations_type]
		equations_true_val = equations_true_by_len_validation[equations_type]
		equations_false_val = equations_false_by_len_validation[equations_type]
		equations_true.extend(equations_true_by_len_train[equations_type + 1])
		equations_false.extend(equations_false_by_len_train[equations_type +
															1])
		equations_true_val.extend(
			equations_true_by_len_validation[equations_type + 1])
		equations_false_val.extend(
			equations_false_by_len_validation[equations_type + 1])
		#the times that the condition of beginning to train MLP is continuously satisfied
		condition_cnt = 0
		while True:
			if equations_type < 1:
				abduced_map = None

			# Abduce and train NN
			consistent_percentage, batch_label_model_precision, abduced_map = abduce_and_train(
				base_model, equations_true, labels, abduced_map, shape,
				SELECT_NUM, NN_BATCHSIZE, NN_EPOCHS)
			if consistent_percentage == 0:
				continue

			#Test if we can use mlp to evaluate
			#The condition is: consistent_percentage >= CONSISTENT_PERCENTAGE_THRESHOLD and batch_label_model_precision>=BATCH_LABEL_MODEL_PRECISION_THRESHOLD
			if consistent_percentage >= CONSISTENT_PERCENTAGE_THRESHOLD and batch_label_model_precision >= BATCH_LABEL_MODEL_PRECISION_THRESHOLD:
				condition_cnt += 1
			else:
				condition_cnt = 0

			#The condition has been satisfied continuously five times
			if condition_cnt >= CONDITION_CNT_THRESHOLD:
				best_accuracy = validation(base_model, equations_true,
										   equations_false, equations_true_val,
										   equations_false_val, labels,
										   abduced_map, shape,
										   LOGIC_OUTPUT_DIM, SAMPLES_PER_RULE,
										   MLP_BATCHSIZE, MLP_EPOCHS)
				# decide next course or restart
				# Save model and go to next course
				if best_accuracy > NEXT_COURSE_ACC_THRESHOLD:
					base_model.save_weights('%s_nlm_weights_%d.hdf5' %
											(src_data_name, equations_type))
					break
				else:
					#Restart current course: reload model
					if equations_type == 0:
						for i in range(len(base_model.layers)):
							base_model.layers[i].set_weights(
								t_model.layers[i].get_weights())
					else:
						base_model.load_weights(
							'%s_nlm_weights_%d.hdf5' %
							(src_data_name, equations_type - 1))
					print("Failed! Reload model.")
					condition_cnt = 0

	#Train final mlp model
	#Calcualte how many equations should be selected in each length
	print("Now begin to train final mlp model")
	select_equation_cnt = []
	if LOGIC_OUTPUT_DIM % EQUATION_LEN_CNT == 0:
		select_equation_cnt = [LOGIC_OUTPUT_DIM // EQUATION_LEN_CNT
							   ] * EQUATION_LEN_CNT
	else:
		select_equation_cnt = [LOGIC_OUTPUT_DIM // EQUATION_LEN_CNT
							   ] * EQUATION_LEN_CNT
		select_equation_cnt[-1] += LOGIC_OUTPUT_DIM % EQUATION_LEN_CNT
	assert sum(select_equation_cnt) == LOGIC_OUTPUT_DIM

	#Abduce rules
	out_rules = []
	#for each length of test equations
	for equations_type in range(EQUATION_LEN_CNT):
		#for each length, there are select_equation_cnt[equations_type] rules
		rules = get_rules_from_data(
			base_model, equations_true_by_len_train[equations_type], labels,
			abduced_map, shape, select_equation_cnt[equations_type],
			SAMPLES_PER_RULE)
		out_rules.extend(rules)

	#Get mlp training data
	mlp_train_vectors, mlp_train_labels = get_all_mlp_data(
		equations_true_by_len_train, equations_false_by_len_train, base_model,
		out_rules, abduced_map, shape, EQUATION_LEN_CNT)
	#Try three times to train and find the best mlp
	for i in range(3):
		#Train MLP
		mlp_model = NN_model.get_mlp_net(LOGIC_OUTPUT_DIM)
		mlp_model.compile(loss='binary_crossentropy',
						  optimizer='rmsprop',
						  metrics=['accuracy'])
		mlp_model.fit(mlp_train_vectors,
					  mlp_train_labels,
					  epochs=MLP_EPOCHS,
					  batch_size=MLP_BATCHSIZE,
					  verbose=0)

		#Test MLP
		for equations_type, (equations_true, equations_false) in enumerate(
				zip(equations_true_by_len_test, equations_false_by_len_test)):
			#for each length of test equations
			mlp_test_len_vectors, mlp_test_len_labels = get_mlp_data(
				equations_true, equations_false, base_model, out_rules,
				abduced_map, shape)
			result = mlp_model.evaluate(mlp_test_len_vectors,
										mlp_test_len_labels,
										batch_size=MLP_BATCHSIZE,
										verbose=0)
			print("The result of testing length %d equations is:" %
				  (equations_type + 5))
			print(result)

	return base_model


def arg_init():
	parser = argparse.ArgumentParser()
	#LOGIC_OUTPUT_DIM = 50 #The mlp vector has 50 dimensions
	parser.add_argument(
		'--LOD',
		dest="LOGIC_OUTPUT_DIM",
		metavar="LOGIC_OUTPUT_DIM",
		type=int,
		default=50,
		help='The last mlp feature vector dimensions, default is 50')

	#EQUATION_MAX_LEN = 8 #Only learn the equations of length 5-8
	parser.add_argument('--EML',
						dest="EQUATION_MAX_LEN",
						metavar='EQUATION_MAX_LEN',
						type=int,
						default=8,
						help='Equation max length in training, default is 8')

	#SELECT_NUM = 10 #Select 10 equations to abduce rules
	parser.add_argument(
		'--SN',
		dest="SELECT_NUM",
		metavar='SELECT_NUM',
		type=int,
		default=10,
		help=
		'Every time pick SELECT_NUM equations to abduce rules, default is 10')

	# Proportion of train and validation = 3:1
	#PROP_TRAIN = 3
	#PROP_VALIDATION = 1
	parser.add_argument(
		'--PT',
		dest="PROP_TRAIN",
		metavar='PROP_TRAIN',
		type=int,
		default=3,
		help='Proportion of train and validation rate, default PROP_TRAIN is 3'
	)
	parser.add_argument(
		'--PV',
		dest="PROP_VALIDATION",
		metavar='PROP_VALIDATION',
		type=int,
		default=1,
		help=
		'Proportion of train and validation rate, default PROP_VALIDATION is 1'
	)

	#CONSISTENT_PERCENTAGE_THRESHOLD = 0.9
	parser.add_argument('--CCT', dest="CONSISTENT_PERCENTAGE_THRESHOLD", metavar='CONSISTENT_PERCENTAGE_THRESHOLD', type=float, default=0.9, \
	  help='Consistent percentage threshold, which decision whether training goes to next stage, default is 0.9')

	#BATCH_LABEL_MODEL_PRECISION_THRESHOLD = 0.9 #If consistent percentage is higher than 0.9 and model precision higher than 0.9, then the condition is satisfied
	parser.add_argument('--BLMPT', dest="BATCH_LABEL_MODEL_PRECISION_THRESHOLD", metavar='BATCH_LABEL_MODEL_PRECISION_THRESHOLD', type=float, default=0.9, \
	  help='If consistent percentage is higher than BATCH_LABEL_MODEL_PRECISION_THRESHOLD and model precision higher than BATCH_LABEL_MODEL_PRECISION_THRESHOLD, then the condition is satisfied, default is 0.9')
	#CONDITION_CNT_THRESHOLD = 5       #If the condition has been satisfied 5 times, the start validation
	parser.add_argument('--CPT', dest="CONDITION_CNT_THRESHOLD", metavar='CONDITION_CNT_THRESHOLD', type=int, default=5, \
	  help='If the condition has been satisfied CONSISTENT_PERCENTAGE_THRESHOLD times, the start validation, default is 5')
	#NEXT_COURSE_ACC_THRESHOLD = 0.86  #If the validation accuracy of a course higher than the threshold, then go to next course
	parser.add_argument('--NCAT', dest="NEXT_COURSE_ACC_THRESHOLD", metavar='NEXT_COURSE_ACC_THRESHOLD', type=float, default=0.86, \
	  help='If the validation accuracy of a course higher than the threshold, then go to next course, default is 0.86')

	#SAMPLES_PER_RULE = 3 # Use 3 samples to abduce a rule when training mlp
	parser.add_argument(
		'--SPR',
		dest="SAMPLES_PER_RULE",
		metavar='SAMPLES_PER_RULE',
		type=int,
		default=3,
		help=
		'Use SAMPLES_PER_RULE samples to abduce a rule when training mlp, default is 3'
	)
	#NN_BATCHSIZE = 32    # Batch size of neural network
	parser.add_argument('--NB',
						dest="NN_BATCHSIZE",
						metavar='NN_BATCHSIZE',
						type=int,
						default=32,
						help='Batch size of neural network, default is 32')
	#NN_EPOCHS = 10       # Epochs of neural network
	parser.add_argument('--NE',
						dest="NN_EPOCHS",
						metavar='NN_EPOCHS',
						type=int,
						default=10,
						help='Epochs of neural network, default is 10')
	#MLP_BATCHSIZE = 128  # Batch size of mlp
	parser.add_argument('--MB',
						dest="MLP_BATCHSIZE",
						metavar='MLP_BATCHSIZE',
						type=int,
						default=128,
						help='Batch size of mlp, default is 128')
	#MLP_EPOCHS = 60      # Epochs of mlp
	parser.add_argument('--ME',
						dest="MLP_EPOCHS",
						metavar='MLP_EPOCHS',
						type=int,
						default=60,
						help='MLP_EPOCHS, default is 60')

	parser.add_argument('--src_dir',
						metavar='dataset dir',
						type=str,
						default="../dataset",
						help="Where store the dataset")
	parser.add_argument('--src_data_name',
						type=str,
						default="mnist_images",
						help="Dataset name")
	parser.add_argument('--height', type=int, default=28, help='Img height')
	parser.add_argument('--weight', type=int, default=28, help='Img weight')
	parser.add_argument('--channel',
						type=int,
						default=1,
						help='Img channel num')
	#parser.add_argument('--pl_file_path', type=str, default="logic/prolog/learn_add.pl", help="Which prolog file will be used")
	parser.add_argument(
		'--src_data_file',
		type=str,
		default="mnist_equation_data_train_len_8_test_len_26_sys_2_.pk",
		help="This file is generated by equation_generator.py")

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	labels = ['0', '1', '10', '11']
	args = arg_init()
	#src_dir = "../dataset"
	src_dir = args.src_dir
	#src_data_name = "mnist_images" #cifar10_images
	src_data_name = args.src_data_name
	#input_shape = (28, 28, 1) #(32, 32, 3)
	input_shape = (args.height, args.weight, args.channel)
	pl_file_path = "logic/prolog/learn_add.pl"
	#pl_file_path = args.pl_file_path
	#src_data_file = 'mnist_equation_data_train_len_8_test_len_26_sys_2_.pk'
	src_data_file = args.src_data_file
	src_path = os.path.join(src_dir, src_data_name)

	net_model_test(src_path=src_path,
				   labels=labels,
				   src_data_name=src_data_name,
				   shape=input_shape)
	net_model_pretrain(src_path=src_path,
					   labels=labels,
					   src_data_name=src_data_name,
					   shape=input_shape)

	model = nlm_main_func(labels=labels,
						  src_data_name=src_data_name,
						  src_data_file=src_data_file,
						  pl_file_path=pl_file_path,
						  shape=input_shape,
						  args=args)

	test_nn_model(model, src_path, labels, input_shape)
