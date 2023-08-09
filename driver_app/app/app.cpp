#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string.h>
#include <time.h>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "../../specification/cpp_implementation/MaxPoolLayer.hpp"
#include "../../specification/cpp_implementation/denselayer.hpp"
#include "../../vp/TLM/addresses.hpp"

#include "bias.hpp"
#include "picture.hpp"
#include "weights0.hpp"
#include "weights1.hpp"
#include "weights2.hpp"

#define IP_COMMAND_LOAD_BIAS			0x0001
#define IP_COMMAND_LOAD_WEIGHTS0		0x0002
#define IP_COMMAND_LOAD_CONV0_INPUT		0x0004
#define IP_COMMAND_START_CONV0			0x0008
#define IP_COMMAND_LOAD_WEIGHTS1		0x0010
#define IP_COMMAND_LOAD_CONV1_INPUT		0x0020
#define IP_COMMAND_START_CONV1			0x0040
#define IP_COMMAND_LOAD_WEIGHTS2		0x0080
#define IP_COMMAND_LOAD_CONV2_INPUT		0x0100
#define IP_COMMAND_START_CONV2			0x0200
#define IP_COMMAND_RESET				0x0400
#define IP_COMMAND_READ_CONV0_OUTPUT	0x0800
#define IP_COMMAND_READ_CONV1_OUTPUT	0x1000
#define IP_COMMAND_READ_CONV2_OUTPUT	0x2000

using namespace std;

typedef vector<vector<vector<vector<float>>>> vector4D;
typedef vector<vector<vector<float>>> vector3D;
typedef vector<vector<float>> vector2D;
typedef vector<float> vector1D;

/* Functions declaration */

void write_ip(int command);
void transform_1D_to_4D(vector1D input_vector, vector4D& output_vector, int img_size, int num_of_channels);
void transform_4D_to_1D(vector4D source_vector,vector1D& dest_vector,int img_size, int num_of_channels);
void flatten(vector4D source_vector,vector2D &dest_vector,int img_size, int num_of_channels);
vector<int> pad_img(vector<int> ram, int img_size, int num_of_channels);
vector<int> format_image(vector<int> ram, int img_size, int num_of_channels);


int main()
{
	vector1D image;
	vector4D image4D;
	vector<int> conv_input;
	vector4D output;
	vector2D dense1_input;
	vector2D dense1_output;
	vector2D dense2_output;
	
	int num_of_pictures = 1;
	int fd;
	int *p;
	
	MaxPoolLayer *maxpool[3];
	DenseLayer *dense_layer[2];
		
	maxpool[0]=new MaxPoolLayer(2);
	maxpool[1]=new MaxPoolLayer(2);
	maxpool[2]=new MaxPoolLayer(2);
	dense_layer[0] = new DenseLayer(1024,512,0);
	dense_layer[1] = new DenseLayer(512,10,1);
	dense_layer[0]->load_dense_layer("../../data/parametars/dense1/dense1_weights.txt", "../../data/parametars/dense1/dense1_bias.txt");
	dense_layer[1]->load_dense_layer("../../data/parametars/dense2/dense2_weights.txt", "../../data/parametars/dense2/dense2_bias.txt");
	
	/* ------------------------ */
	/* Send biases at the start */
	/* ------------------------ */
	write_ip(IP_COMMAND_LOAD_BIAS);

	
	fd = open("/dev/cnn-ip", O_RDWR | O_NDELAY);
	if(fd < 0)
	{
		cout << "[app] Cannot open /dev/cnn-ip for write" << endl;
		return -1;
	}
	
	p = (int*)mmap(0, 128*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	memcpy(p, input_bias, 128*2);
	munmap(p, 128*2);
	close(fd);
	if (fd < 0)
	{
		cout << "[app] Cannot close /dev/dma for write" << endl;
		return -1;
	}
	usleep(50000);
	
	cout << "Biases loaded\n" << endl;
	
	
	/* ------------------------ */
	/* -----Classification----- */
	/* ------------------------ */
	
	for(int picture = 0; picture < num_of_pictures; picture++)
	{
		cout << "Starting classification..." << endl;
		
		/* Send weights0 */
	/*	
		write_ip(IP_COMMAND_LOAD_WEIGHTS0);
		
		p = (int*)mmap(0, 864*4, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(p, input_weights0, 864*4);
		munmap(p, 864*4);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		usleep(50000);
	*/	
		
		/* Send picture */
	/*	
		write_ip(IP_COMMAND_LOAD_CONV0_INPUT);
		
		p = (int*)mmap(0, 3468*4, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(p, picture_0, 3468*4);
		munmap(p, 3468*4);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		usleep(50000);
	*/	
		
		/* Start CONV0 */
	/*	
		write_ip(IP_COMMAND_START_CONV0);
		
		usleep(100000);
	*/	
		
		/* Read results */
	/*
		write_ip(IP_COMMAND_READ_CONV0_OUTPUT);
		
		p = (int*)mmap(0, 32768*4, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(, , 32768*4);
		munmap(p, 32768*4);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		usleep(50000);
	*/
		
		/* Maxpool for CONV0 output */
	/*
		transform_1D_to_4D(image, image4D, CONV1_PICTURE_SIZE, CONV1_NUM_FILTERS);
		output.clear();
		output = maxpool[0]->forward_prop(image4D, {}); 

		transform_4D_to_1D(output, image, CONV1_PICTURE_SIZE/2, CONV1_NUM_FILTERS);
		
		// Transforming vector<float> to vector<int>
		conv_input.clear();
		for (long unsigned int i = 0; i < image.size(); ++i)
		{
			conv_input.push_back(image[i]);
		}
		
		conv_input = pad_img(conv_input, CONV2_PICTURE_SIZE, CONV2_NUM_CHANNELS);

		conv_input = format_image(conv_input, CONV2_PADDED_PICTURE_SIZE, CONV2_NUM_CHANNELS);
		
		
	*/
	}
	
	
	return 0;
}


void write_ip(int command)
{
	FILE *cnn_file;
	cnn_file = fopen("/dev/cnn-ip", "w");
	if(cnn_file < 0) cout << "[app] Could not open /dev/cnn-ip" << endl;
	else cout << "[app] Opened /dev/cnn-ip" << endl;

	fprintf(cnn_file, "%d\n", command);
	if(fclose(cnn_file)) cout << "[app] Cannot close /dec/cnn-ip" << endl;
	else cout << "[app] Successfully closed /dev/cnn-ip" << endl;
}

void flatten(vector4D source_vector,vector2D &dest_vector,int img_size, int num_of_channels)
{
	dest_vector.clear();
	vector1D tmp;
	for (int row = 0; row < img_size; ++row)
	{	
		for (int column = 0; column < img_size; ++column)
		{
			for (int channel = 0; channel < num_of_channels; ++channel)
			{
				tmp.push_back(source_vector[0][row][column][channel]);
			}
		}
	}
	dest_vector.push_back(tmp);
}

void transform_1D_to_4D(vector1D input_vector, vector4D& output_vector, int img_size, int num_of_channels)
{
	output_vector.clear();
	vector3D rows;
	for (int row = 0; row < img_size; ++row)
	{	
		vector2D columns;
		for (int column = 0; column < img_size; ++column)
		{
			vector1D channels; 
			for (int channel = 0; channel < num_of_channels; ++channel)
			{
				channels.push_back(input_vector[channel * img_size*img_size + column + row * img_size]);
			}
			columns.push_back(channels);
		}
		rows.push_back(columns);
	}
	output_vector.push_back(rows);
}

void transform_4D_to_1D(vector4D source_vector, vector1D& dest_vector,int img_size, int num_of_channels)
{
	dest_vector.clear();
	for (int channel = 0; channel < num_of_channels; ++channel)
	{	
		for (int row = 0; row < img_size; ++row)
		{
			for (int column = 0; column < img_size; ++column)
			{
				dest_vector.push_back(source_vector[0][row][column][channel]);
			}
		}
	}
}

vector<int> pad_img(vector<int> ram, int img_size, int num_of_channels)
{
	for(int channel = 0 ; channel < num_of_channels; channel++)
    	{
			// Firstly, zeros are emplaced for the first padded row (image_size(one row) + 2 for the edges)
	        for (int i = 0; i < img_size+2; i++)
	        {
	        	ram.emplace((ram.begin() + (channel)*(img_size+2)*(img_size+2) + i), 0);
	        }

			// Secondly, zeros are added to each row's edge
	        for(int rows = 1; rows < img_size + 1; rows++)
	        {
				// pos1 calulates the position to insert the left-most zero in each row (left edge)
				// Component "(channel)*(img_size+2)*(img_size+2)" refers to the size of the channel
				// Component "rows*img_size" refers to the number of rows that have been padded on the current channel
				// Component "rows*2" takes into account the number of edge pixels that have been added (padded) on the current channel
	        	int pos1 = (channel)*(img_size+2)*(img_size+2) + rows*img_size + rows*2;
				// pos2 calulates the position to insert the right-most zero in each row (right edge)
	        	int pos2 = (channel)*(img_size+2)*(img_size+2) + rows*img_size + rows*2 + 1 + img_size;
	        	ram.emplace((ram.begin() + pos1), 0);
	        	ram.emplace((ram.begin() + pos2), 0);
	        }

			// Finally, zeros are pushed back as we fill up the final row of the padded image (plus 2 for edges)
	        for (int i = 0; i < img_size + 2; i++)
	        {
	        	ram.emplace((ram.begin() + ((channel)*(img_size+2)*(img_size+2)) + (img_size+2)*(img_size+1) + i), 0);
	        }
    	}
    	
    	return ram;
}

vector<int> format_image(vector<int> ram, int img_size, int num_of_channels)
{
	vector <int> temp_ram;

	temp_ram.clear();
	
	for(int i = 0; i < img_size; i++)
	{
		for(int j = 0; j < num_of_channels; j++)
		{
			for(int k = 0; k < 3; k++)
			{
				temp_ram.push_back(ram[i + j * img_size * img_size + k * img_size]);
			}
		}
	}
	
	for(int i = 3; i < img_size; i++)
	{
		for(int j = 0; j < img_size; j++)
		{
			for(int k = 0; k < num_of_channels; k++)
			{
				temp_ram.push_back(ram[j + k * img_size * img_size + i * img_size]);
			}
		}
	}

	ram.clear();
	//ofstream results_file;
	//results_file.open("../../data/picture1_formated.txt");
	for(int i = 0; i < temp_ram.size(); i++) 
	{
		ram.push_back(temp_ram[i]);
		//results_file << ram[ram.size() - 1] << endl;


	}

	//results_file.close();
	return ram;
}
