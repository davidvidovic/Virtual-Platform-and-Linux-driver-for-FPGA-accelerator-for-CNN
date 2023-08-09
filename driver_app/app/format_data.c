#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int castFloatToBin(float val);
float castBinToFloat(int binaryValue);

int main()
{
	/* Formating bias */
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
	
	/* ----------------------------------------- */
	/* Formatting weights0 */
	
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
	
	/* ----------------------------------------- */
	/* Formatting weights1 */
	
	fp = fopen("weights1.txt", "w");
	input = fopen("../../data/conv1_input/weights1_formated.txt", "r");
	
	for(int i = 0; i < 9216; i++)
	{
		fscanf(input, "%f", &temp);
		res = castFloatToBin(temp);
		fprintf(fp, "%d,", res);
		if(i%32 == 0) fprintf(fp, "\n");
	}
	
	fclose(fp);
	fclose(input);
	
	printf("Weights1 done\n");
	
	/* ----------------------------------------- */
	/* Formatting weights2 */
	
	fp = fopen("weights2.txt", "w");
	input = fopen("../../data/conv2_input/weights2_formated.txt", "r");
	
	for(int i = 0; i < 18432; i++)
	{
		fscanf(input, "%f", &temp);
		res = castFloatToBin(temp);
		fprintf(fp, "%d,", res);
		if(i%32 == 0) fprintf(fp, "\n");
	}
	
	fclose(fp);
	fclose(input);
	
	printf("Weights2 done\n");
	
	/* ----------------------------------------- */
	/* Formatting picture0 */
	
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
