import numpy as np
import os
import matplotlib.pyplot as plt


def seize_result(root_folder, algo_name):
	D = np.load(os.path.join(root_folder, 'test_score.npy'))

	# print('Highest accuracy ' + str(D.max()))
	return algo_name, D.max()


if __name__ == "__main__":
	dataset = 'somethingV1'

	if dataset == 'Kinetics400':
		root_folder = "../data_ckpt/kinetics400_ckpt/"
	elif dataset == 'somethingV1':
		root_folder = "../data_ckpt/somethingV1_ckpt/"
	elif dataset == 'UCF101':
		root_folder = "../data_ckpt/UCF101_ckpt/"
	else:
		raise ValueError('Unknown dataset ' + dataset)

	result_list = []

	folder_list = os.listdir(root_folder)
	for folder in folder_list:
		folder_path = os.path.join(root_folder, folder)
		# algo_name = folder.split('_')[1]
		algo_name = folder

		subfolder_list = os.listdir(folder_path)
		for subfolder in subfolder_list:
			subfolder_path = os.path.join(folder_path, subfolder)
			if os.path.isdir(subfolder_path):
				precised_algo_name = algo_name + '+' + subfolder
				result_list.append(seize_result(subfolder_path, precised_algo_name))
			else:
				result_list.append(seize_result(folder_path, algo_name))
				break

	result_list.sort(key=lambda result: result[1])

	print('\n')
	for result in result_list:
		print('{0:50}{1:50}'.format(result[0], result[1]))


