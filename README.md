
Welcome to the python tools repo for working with phase-domain spiking neural networks

At the moment this tool is a single-line operation:
    python toolchain.py <sample_file_name> <target_file_name> <bitstream_destination>

=================================================
This command takes labeled data in the form of input samples and corresponding targets. 
Pass the name of the file into the respective field. Sample and target data must be files that 
were created with torch.save(). Each should contain a single pytorch tensor. 

An example of how sample data for an xor should be saved:

inputs = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
torch.save(inputs, 'inputs') 

=================================================

In its current form the toolchain (1) trains a model on the labeled data, (2) refits for SNN hardware, 
and (3) generates a bitstream for the weights. This bitstream is stored at <bitstream_destination> as a csv. 
The first element is LSB of input 1 into the top left neuron (as seen from Taylor's most recent diagram) 

===============================================

FUTURE DEVELOPMENT: 

Here are some functionalities that would improve these tools greatly. New developers should start with these. 
I will also be working on these as my schedule permits: 

-Dividing the tool into three stages (pytorch model->hardware simulation->bitstream) accessible to the user

-A simple command-line gui

-Modifying the learning algorithm for time-series data 

-toggled parameters that tie chip outputs back to input for feedback (would help with temporal learning) 

-optimizations to the phase-domain neuron model

-more graphing utilities 

-weights have a bitstream, does the toolchain need to export anything for vco biases?

-separate config file, can set learning hyperparameters and also capacitance/voltage data for the pdn model

*add wanted features here* 










