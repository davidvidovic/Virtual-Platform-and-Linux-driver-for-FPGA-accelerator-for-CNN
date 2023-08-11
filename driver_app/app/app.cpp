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
#include <cstdint>

#include "../../specification/cpp_implementation/MaxPoolLayer.hpp"
#include "../../specification/cpp_implementation/denselayer.hpp"
#include "../../vp/TLM/addresses.hpp"
#include "format_data.hpp"

#include "bias.hpp"
#include "picture.hpp"
#include "weights0.hpp"
#include "weights1_0.hpp"
#include "weights1_1.hpp"
#include "weights2_0.hpp"
#include "weights2_1.hpp"
#include "weights2_2.hpp"
#include "weights2_3.hpp"

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

void write_ip(int command);

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
	uint16_t temp;
	float fl_temp;
	uint16_t picture1[10368];
	uint16_t picture2[6400];

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
	/* --------Reset IP-------- */
	/* ------------------------ */

	write_ip(IP_COMMAND_RESET);
	usleep(5000);	

	/* ------------------------ */
	/* Send biases at the start */
	/* ------------------------ */

	fd = open("/dev/dma", O_RDWR | O_NDELAY);
	if(fd < 0)
	{
		cout << "[app] Cannot open /dev/dma for write" << endl;
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

	write_ip(1);
	usleep(50000);
	
	cout << "Biases loaded\n" << endl;
	
	
	/* ------------------------ */
	/* -----Classification----- */
	/* ------------------------ */
	
	for(int picture = 0; picture < num_of_pictures; picture++)
	{
		cout << "Starting classification..." << endl;
		
		/* Send weights0 */
		
		fd = open("/dev/dma", O_RDWR | O_NDELAY);
		if(fd < 0)
		{
			cout << "[app] Cannot open /dev/dma for write" << endl;
			return -1;
		}
		
		p = (int*)mmap(0, 864*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(p, input_weights0, 864*2);
		munmap(p, 864*2);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		
		write_ip(IP_COMMAND_LOAD_WEIGHTS0);
		usleep(50000);
		
		
		/* Send input picture to CONV0 */
		
		
		fd = open("/dev/dma", O_RDWR | O_NDELAY);
		if(fd < 0)
		{
			cout << "[app] Cannot open /dev/dma for write" << endl;
			return -1;
		}
		p = (int*)mmap(0, 3468*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(p, picture_0, 3468*2);
		munmap(p, 3468*2);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		
		write_ip(IP_COMMAND_LOAD_CONV0_INPUT);
		usleep(50000);
	
		
		/* Start CONV0 */
	
		write_ip(IP_COMMAND_START_CONV0);
		
		usleep(1000000);
		
		
		/* Read results */
	
		write_ip(IP_COMMAND_READ_CONV0_OUTPUT);
		usleep(50000);	
		fd = open("/dev/dma", O_RDWR | O_NDELAY);
		if(fd < 0)
		{
			cout << "[app] Cannot open /dev/dma for write" << endl;
			return -1;
		}
		p = (int *)mmap(0, 32768*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if(p == MAP_FAILED) cout << "MAP FAILED" << endl;
		
/*
 		int cnt = 1;
		for(int i = 0; i < 50; i++)
		{
			cout << cnt++ << ": " << (uint16_t)((uint32_t)*(p+i) & 0x0000ffff) << endl;
			cout << cnt++ << ": " << (uint16_t)(((uint32_t)*(p+i) & 0xffff0000) >> 16)	<< endl;
		}
*/
		
		image.clear();

		for(int i = 0; i < 32768*2; i++)
		{
			temp = (uint16_t)((uint32_t)*(p+i) & 0x0000ffff);
			fl_temp = castBinToFloat((int)temp);
			image.push_back(fl_temp);

			temp = (uint16_t)(((uint32_t)*(p+i) & 0xffff0000) >> 16);
			fl_temp = castBinToFloat((int)temp);
			image.push_back(fl_temp);
		}

		munmap(p, 32768*2);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		
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

		for(int i = 0; i < conv_input.size(); i++) picture_1[i] = castFloatToBin(conv_input[i]);
*/	
		

		/* Send input picture to CONV1 */
		
/*
		fd = open("/dev/dma", O_RDWR | O_NDELAY);
		if(fd < 0)
		{
			cout << "[app] Cannot open /dev/dma for write" << endl;
			return -1;
		}
		p = (int*)mmap(0, 10368*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(p, picture_1, 10368*2);
		munmap(p, 10368*2);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		
		write_ip(IP_COMMAND_LOAD_CONV1_INPUT);
		usleep(50000);
*/


		/* Send 1/2 of weights1 */
/*		
		fd = open("/dev/dma", O_RDWR | O_NDELAY);
		if(fd < 0)
		{
			cout << "[app] Cannot open /dev/dma for write" << endl;
			return -1;
		}
		
		p = (int*)mmap(0, 4608*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(p, input_weights1_0, 4608*2);
		munmap(p, 4608*2);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		
		write_ip(IP_COMMAND_LOAD_WEIGHTS1);
		usleep(50000);
*/

		/* Start 1/2 of CONV1 */
/*
		write_ip(IP_COMMAND_START_CONV1);
		usleep(50000);
*/

		/* Send 2/2 of weights1 */
/*		
		fd = open("/dev/dma", O_RDWR | O_NDELAY);
		if(fd < 0)
		{
			cout << "[app] Cannot open /dev/dma for write" << endl;
			return -1;
		}
		
		p = (int*)mmap(0, 4608*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(p, input_weights1_1, 4608*2);
		munmap(p, 4608*2);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		
		write_ip(IP_COMMAND_LOAD_WEIGHTS1);
		usleep(50000);
*/

		/* Start 2/2 of CONV1 */
/*
		write_ip(IP_COMMAND_START_CONV1);
		usleep(50000);
*/

		/* Read results */
/*	
		write_ip(IP_COMMAND_READ_CONV1_OUTPUT);
		usleep(50000);	
		fd = open("/dev/dma", O_RDWR | O_NDELAY);
		if(fd < 0)
		{
			cout << "[app] Cannot open /dev/dma for write" << endl;
			return -1;
		}
		p = (int *)mmap(0, 8192*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if(p == MAP_FAILED) cout << "MAP FAILED" << endl;
	

		image.clear();

		for(int i = 0; i < 8192*2; i++)
		{
			temp = (uint16_t)((uint32_t)*(p+i) & 0x0000ffff);
			fl_temp = castBinToFloat((int)temp);
			image.push_back(fl_temp);

			temp = (uint16_t)(((uint32_t)*(p+i) & 0xffff0000) >> 16);
			fl_temp = castBinToFloat((int)temp);
			image.push_back(fl_temp);
		}

		munmap(p, 8192*2);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
*/

		/* Maxpool for CONV1 output */
/*
		transform_1D_to_4D(image, image4D, CONV2_PICTURE_SIZE, CONV2_NUM_FILTERS);
		output.clear();
		output = maxpool[1]->forward_prop(image4D, {}); 

		transform_4D_to_1D(output, image, CONV2_PICTURE_SIZE/2, CONV2_NUM_FILTERS);
		
		// Transforming vector<float> to vector<int>
		conv_input.clear();
		for (long unsigned int i = 0; i < image.size(); ++i)
		{
			conv_input.push_back(image[i]);
		}
		
		conv_input = pad_img(conv_input, CONV3_PICTURE_SIZE, CONV3_NUM_CHANNELS);

		conv_input = format_image(conv_input, CONV3_PADDED_PICTURE_SIZE, CONV3_NUM_CHANNELS);

		for(int i = 0; i < conv_input.size(); i++) picture_2[i] = castFloatToBin(conv_input[i]);
*/	

		/* Send input picture to CONV2 */
		
/*
		fd = open("/dev/dma", O_RDWR | O_NDELAY);
		if(fd < 0)
		{
			cout << "[app] Cannot open /dev/dma for write" << endl;
			return -1;
		}
		p = (int*)mmap(0, 6400*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(p, picture_2, 6400*2);
		munmap(p, 6400*2);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		
		write_ip(IP_COMMAND_LOAD_CONV2_INPUT);
		usleep(50000);
*/


		/* Send 1/4 of weights2 */
/*		
		fd = open("/dev/dma", O_RDWR | O_NDELAY);
		if(fd < 0)
		{
			cout << "[app] Cannot open /dev/dma for write" << endl;
			return -1;
		}
		
		p = (int*)mmap(0, 4608*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(p, input_weights2_0, 4608*2);
		munmap(p, 4608*2);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		
		write_ip(IP_COMMAND_LOAD_WEIGHTS2);
		usleep(50000);
*/

		/* Start 1/4 of CONV2 */
/*
		write_ip(IP_COMMAND_START_CONV2);
		usleep(50000);
*/

		/* Send 2/4 of weights2 */
/*		
		fd = open("/dev/dma", O_RDWR | O_NDELAY);
		if(fd < 0)
		{
			cout << "[app] Cannot open /dev/dma for write" << endl;
			return -1;
		}
		
		p = (int*)mmap(0, 4608*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(p, input_weights2_1, 4608*2);
		munmap(p, 4608*2);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		
		write_ip(IP_COMMAND_LOAD_WEIGHTS2);
		usleep(50000);
*/

		/* Start 2/4 of CONV2 */
/*
		write_ip(IP_COMMAND_START_CONV2);
		usleep(50000);
*/


		/* Send 3/4 of weights2 */
/*		
		fd = open("/dev/dma", O_RDWR | O_NDELAY);
		if(fd < 0)
		{
			cout << "[app] Cannot open /dev/dma for write" << endl;
			return -1;
		}
		
		p = (int*)mmap(0, 4608*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(p, input_weights2_2, 4608*2);
		munmap(p, 4608*2);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		
		write_ip(IP_COMMAND_LOAD_WEIGHTS2);
		usleep(50000);
*/

		/* Start 3/4 of CONV2 */
/*
		write_ip(IP_COMMAND_START_CONV2);
		usleep(50000);
*/

		/* Send 4/4 of weights2 */
/*		
		fd = open("/dev/dma", O_RDWR | O_NDELAY);
		if(fd < 0)
		{
			cout << "[app] Cannot open /dev/dma for write" << endl;
			return -1;
		}
		
		p = (int*)mmap(0, 4608*2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		memcpy(p, input_weights2_3, 4608*2);
		munmap(p, 4608*2);
		close(fd);
		if (fd < 0)
		{
			cout << "Cannot close /dev/dma for write" << endl;
			return -1;
		}
		
		write_ip(IP_COMMAND_LOAD_WEIGHTS2);
		usleep(50000);
*/

		/* Start 4/4 of CONV2 */
/*
		write_ip(IP_COMMAND_START_CONV2);
		usleep(50000);
*/

		/* Maxpool for CONV2 output */
/*
		transform_1D_to_4D(image, image4D, CONV3_PICTURE_SIZE, CONV3_NUM_FILTERS);
		output.clear();
		output = maxpool[2]->forward_prop(image4D, {}); 

		flatten(output,dense1_input,CONV3_PICTURE_SIZE/2,CONV3_NUM_FILTERS);

		dense1_output=dense_layer[0]->forward_prop(dense1_input);
		dense2_output=dense_layer[1]->forward_prop(dense1_output);

		cout << "Picture 0 results: " << endl;
		for (int i = 0; i < 10; ++i)
		{
			cout << dense2_output[0][i] << endl;
		}
*/

	}
	
	
	return 0;
}


void write_ip(int command)
{
	FILE *cnn_file;
	cnn_file = fopen("/dev/cnn-ip", "w");
	if(cnn_file == NULL) cout << "[app] Could not open /dev/cnn-ip" << endl;
	else cout << "[app] Opened /dev/cnn-ip" << endl;

	fprintf(cnn_file, "%d\n", command);
	if(fclose(cnn_file)) cout << "[app] Cannot close /dev/cnn-ip" << endl;
	else cout << "[app] Successfully closed /dev/cnn-ip" << endl;
}
