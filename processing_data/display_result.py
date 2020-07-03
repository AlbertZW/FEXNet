import numpy as np
import os
import matplotlib.pyplot as plt


def display_data(root_folder):
	A = np.load(os.path.join(root_folder, 'training_losses.npy'))
	B = np.load(os.path.join(root_folder, 'training_scores.npy'))

	C = np.load(os.path.join(root_folder, 'test_loss.npy'))
	D = np.load(os.path.join(root_folder, 'test_score.npy'))

	epochs = A.size

	print('Highest accuracy ' + str(D.max()) + 'at epoch ' + str(np.argmax(D) * 5))

	# plot
	fig = plt.figure(figsize=(10, 4))
	plt.subplot(121)
	plt.plot(np.arange(1, epochs + 1), A)  # train loss (on epoch end)
	plt.plot(np.arange(5, epochs + 1, 5), C)  # test loss (on epoch end)
	plt.title("model loss")
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend(['train', 'test'], loc="upper left")
	# 2nd figure
	plt.subplot(122)
	plt.plot(np.arange(1, epochs + 1), B)  # train accuracy (on epoch end)
	plt.plot(np.arange(5, epochs + 1, 5), D)  # test accuracy (on epoch end)
	plt.title("training scores")
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend(['train', 'test'], loc="upper left")
	title = os.path.join(root_folder, "fig_TFN.png")
	plt.savefig(title, dpi=600)
	# plt.close(fig)
	plt.show()

if __name__ == "__main__":
	# dataset = 'somethingV1'
	dataset = 'Kinetics400'

	if dataset == 'Kinetics400':
		root_folder = "../data_ckpt/kinetics400_ckpt/"
	elif dataset == 'somethingV1':
		root_folder = "../data_ckpt/somethingV1_ckpt/"
	elif dataset == 'UCF101':
		root_folder = "../data_ckpt/UCF101_ckpt/"
	else:
		raise ValueError('Unknown dataset ' + dataset)

	display_data(root_folder)
