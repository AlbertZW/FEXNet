class argSet:
	def __init__(self, dataset, segment_num, dropout, arch, gpus,
	             epochs, train_batch_size, val_batch_size, lr_steps, resume=False, evaluate=False,
	             applyPretrainedModel=False, modality='RGB', flow_prefix='flow_'):
		self.dataset = dataset
		self.num_segments = segment_num
		self.dropout = dropout
		self.arch = arch
		self.gpus = gpus
		self.epochs = epochs
		self.train_batch_size = train_batch_size
		self.val_batch_size = val_batch_size
		self.lr_steps = lr_steps

		self.resume = resume
		self.evaluate = evaluate
		self.applyPretrainedModel = applyPretrainedModel
		self.modality = modality
		self.flow_prefix = flow_prefix


		self.lr = 0.001
		self.clip_gradient = 20
		self.print_freq = 20
		self.eval_freq = 5
		self.workers = 4
		self.train_list = 'trainlist.txt'
		self.val_list = 'testlist.txt'
		self.loss_type = 'nll'
		self.momentum = 0.9
		self.weight_decay = 5e-4

class DatasourceConfig:
	def __init__(self, dataset, modality):
		if dataset == 'Kinetics400':
			data_path = "/data/Disk_C/Kinetics400_Resources/"
			# data_path = '/media/albert/DATA/_DataSources/Kinetics400/'
			if modality == 'Flow':
				pass
			list_folder = './data_ckpt/kinetics400'
			save_model_path = "./data_ckpt/kinetics400_ckpt/"
			frame_format = 'frame_{:04d}.jpg'
			num_class = 400  # number of target category

		elif dataset == 'somethingV1':
			data_path = '/data/Disk_C/something/20bn-something-something-v1'
			if modality == 'Flow':
				pass
			list_folder = './data_ckpt/somethingV1'
			save_model_path = "./data_ckpt/somethingV1_ckpt/"
			frame_format = '{:05d}.jpg'
			num_class = 174

		elif dataset == 'somethingV2':
			data_path = '/data/Disk_C/something/20bn-something-something-v2_images'
			if modality == 'Flow':
				pass
			list_folder = './data_ckpt/somethingV2'
			save_model_path = "./data_ckpt/somethingV2_ckpt/"
			frame_format = '{:04d}.jpg'
			num_class = 174

		elif dataset == 'UCF101':
			# data_path = '/data/Disk_C/UCF101_Resources/UCF-101_IMAGES/'
			# data_path = '/media/albert/DATA1/_DataSources/UCF-101_IMAGES/'
			data_path = 'D:\\_Datasets\\UCF-101_IMAGES'
			if modality == 'Flow':
				pass
			list_folder = './data_ckpt/UCF101'
			save_model_path = "./data_ckpt/UCF101_ckpt/"
			frame_format = 'frame_{:03d}.jpg'
			num_class = 101

		else:
			raise ValueError('Unknown dataset ' + dataset)


		self.data_path = data_path
		self.list_folder = list_folder
		self.save_model_path = save_model_path
		self.frame_format = frame_format
		self.num_class = num_class

