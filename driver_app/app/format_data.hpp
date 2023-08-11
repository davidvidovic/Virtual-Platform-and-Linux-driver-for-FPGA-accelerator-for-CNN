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
