import os.path
import torch

'''
checkpointing source:
https://blog.floydhub.com/checkpointing-tutorial-for-tensorflow-keras-and-pytorch/
'''
def save_checkpoint(state, checkpoint_dir):
	filename = checkpoint_dir + '/network.pth.tar'
	print "Saving checkpoint for epoch " + str(epoch) + " at " + filename + " ..."
	torch.save(state, filename)  # save checkpoint
	print "Saved checkpoint."

def get_checkpoint(checkpoint_dir, epoch):
	resume_weights = checkpoint_dir + '/epoch' + str(epoch) + '.pth.tar'
	if torch.cuda.is_available():
		print "Attempting to load Cuda weights..."
		checkpoint = torch.load(resume_weights)
		print "Loaded weights."
	else:
		print "Attempting to load weights for CPU..."
		# Load GPU model on CPU
		checkpoint = torch.load(resume_weights,
								map_location=lambda storage,
								loc: storage)
		print "Loaded weights."
	return checkpoint

def int_tensor(input):
	if torch.cuda.is_available():
		return torch.cuda.IntTensor(input)
	else:
		return torch.IntTensor(input)

def float_tensor(input):
	if torch.cuda.is_available():
		return torch.cuda.FloatTensor(input)
	else:
		return torch.FloatTensor(input)

def perform_no_ops(ale, no_op_max, preprocessor, state, random_state):
	#perform nullops
	num_no_ops = random_state.randint(1, no_op_max + 1)
	for _ in range(num_no_ops):
		ale.act(0)
		preprocessor.add(ale.getScreenRGB())
	if len(preprocessor.preprocess_stack) < 2:
		ale.act(0)
		preprocessor.add(ale.getScreenRGB())
	state.add_frame(preprocessor.preprocess())
