#include "format_data.hpp"

using namespace std;

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


int castFloatToBin(float val)
{
	int sign;
	int integerPart;
	int decimalPart;
	int binary;

	sign = ((val>=0) ? 0 : 1);

	  if (sign == 0)
	  {
		integerPart = (int) val;
		decimalPart = (int) ((val - integerPart)*4096);
		binary = (sign << 15) | ((integerPart & 0x7) << 12) | (decimalPart & 0xFFF);
	  }
	  else
	  {
		integerPart = (int) ((-1)*val);
		decimalPart = (int)(((-1)*val - integerPart)*4096);
		binary = (sign << 15) | (((~integerPart) & 0x7) << 12) | ((~decimalPart) & 0xFFF);
		binary = binary + 0b000000000001;
	  }
	  return binary;

}

float castBinToFloat(int binaryValue) {
    int binaryValue_uint = binaryValue;
    int sign = (binaryValue_uint >> 15) & 0x1;

    if (sign == 1) {
        binaryValue_uint = (~binaryValue_uint) + 1; // prebacujemo u pozitivno, posle cemo float pomnozitit sa -1
    }

    int integerPart = (binaryValue_uint >> 12) & 0x7;
    int decimalPart = binaryValue_uint & 0xFFF;

    float floatValue = (float)integerPart + ((float)decimalPart / 4096.0f);
    if (sign == 1)
        floatValue = floatValue * (-1);

    return floatValue;
}

/*
int main()
{
	// Formating bias 
	FILE *fp;
	fp = fopen("biases.txt", "w");
	
	FILE *input;
	input = fopen("../../data/conv0_input/bias_formated.txt", "r");
	
	float temp;
	int res;
	
	for(int i = 0; i < 128; i++)
	{
		fscanf(input, "%f", &temp);
		res = castFloatToBin(temp);
		fprintf(fp, "%d,", res);
		if(i%32 == 0) fprintf(fp, "\n");
	}
	
	fclose(fp);
	fclose(input);
	
	printf("Bias done\n");
	
	// ----------------------------------------- 
	// Formatting weights0
	
	fp = fopen("weights0.txt", "w");
	input = fopen("../../data/conv0_input/weights0_formated.txt", "r");
	
	for(int i = 0; i < 864; i++)
	{
		fscanf(input, "%f", &temp);
		res = castFloatToBin(temp);
		fprintf(fp, "%d,", res);
		if(i%32 == 0) fprintf(fp, "\n");
	}
	
	fclose(fp);
	fclose(input);
	
	printf("Weights0 done\n");
	
	// ----------------------------------------- 
	// Formatting weights1
	
	fp = fopen("weights1_0.txt", "w");
	input = fopen("../../data/conv1_input/weights1_formated.txt", "r");
	
	for(int i = 0; i < 4608; i++)
	{
		fscanf(input, "%f", &temp);
		res = castFloatToBin(temp);
		fprintf(fp, "%d,", res);
		if(i%32 == 0) fprintf(fp, "\n");
	}
	
	fclose(fp);
	fclose(input);

	fp = fopen("weights1_1.txt", "w");
	input = fopen("../../data/conv1_input/weights1_formated.txt", "r");
	
	for(int i = 0; i < 9216; i++)
	{
		fscanf(input, "%f", &temp);
		if(i > 4608)
		{
			fscanf(input, "%f", &temp);
			res = castFloatToBin(temp);
			fprintf(fp, "%d,", res);
			if(i%32 == 0) fprintf(fp, "\n");
		}
	}
	
	fclose(fp);
	fclose(input);
	
	printf("Weights1 done\n");
	
	// ----------------------------------------- 
	// Formatting weights2 
	
	fp = fopen("weights2_0.txt", "w");
	input = fopen("../../data/conv2_input/weights2_formated.txt", "r");
	
	for(int i = 0; i < 4608; i++)
	{
		fscanf(input, "%f", &temp);
		res = castFloatToBin(temp);
		fprintf(fp, "%d,", res);
		if(i%32 == 0) fprintf(fp, "\n");
	}
	
	fclose(fp);
	fclose(input);



	fp = fopen("weights2_1.txt", "w");
	input = fopen("../../data/conv2_input/weights2_formated.txt", "r");
	
	for(int i = 0; i < 9216; i++)
	{
		fscanf(input, "%f", &temp);
		if(i > 4608)
		{
			fscanf(input, "%f", &temp);
			res = castFloatToBin(temp);
			fprintf(fp, "%d,", res);
			if(i%32 == 0) fprintf(fp, "\n");
		}
	}
	
	fclose(fp);
	fclose(input);




	fp = fopen("weights2_2.txt", "w");
	input = fopen("../../data/conv2_input/weights2_formated.txt", "r");
	
	for(int i = 0; i < 13824; i++)
	{
		fscanf(input, "%f", &temp);
		if(i > 9216)
		{
			fscanf(input, "%f", &temp);
			res = castFloatToBin(temp);
			fprintf(fp, "%d,", res);
			if(i%32 == 0) fprintf(fp, "\n");
		}
	}
	
	fclose(fp);
	fclose(input);




	fp = fopen("weights2_3.txt", "w");
	input = fopen("../../data/conv2_input/weights2_formated.txt", "r");
	
	for(int i = 0; i < 18432; i++)
	{
		fscanf(input, "%f", &temp);
		if(i > 13824)
		{
			fscanf(input, "%f", &temp);
			res = castFloatToBin(temp);
			fprintf(fp, "%d,", res);
			if(i%32 == 0) fprintf(fp, "\n");
		}
	}
	
	fclose(fp);
	fclose(input);
	
	printf("Weights2 done\n");
	
	// ----------------------------------------- 
	// Formatting picture0 
	
	fp = fopen("picture.txt", "w");
	input = fopen("../../data/conv0_input/picture_conv0_input.txt", "r");
	
	for(int i = 0; i < 3468; i++)
	{
		fscanf(input, "%f", &temp);
		res = castFloatToBin(temp);
		fprintf(fp, "%d,", res);
		if(i%32 == 0) fprintf(fp, "\n");
	}
	
	fclose(fp);
	fclose(input);
	
	printf("Picture done\n");
	
	return 0;
}
*/


