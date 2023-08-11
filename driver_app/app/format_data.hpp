#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

typedef std::vector<std::vector<std::vector<std::vector<float>>>> vector4D;
typedef std::vector<std::vector<std::vector<float>>> vector3D;
typedef std::vector<std::vector<float>> vector2D;
typedef std::vector<float> vector1D;

int castFloatToBin(float val);
float castBinToFloat(int binaryValue);
void flatten(vector4D source_vector, vector2D &dest_vector, int img_size, int num_of_channels);
void transform_1D_to_4D(vector1D input_vector, vector4D& output_vector, int img_size, int num_of_channels);
void transform_4D_to_1D(vector4D input_vector, vector1D& output_vector, int img_size, int num_of_channels);
std::vector<int> format_image(std::vector<int> ram, int img_size, int num_of_channels);
std::vector<int> pad_img(std::vector<int> ram, int img_size, int num_of_channels);
