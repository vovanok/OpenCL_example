// �������� ������������ ���� OpenCL
#include <CL/cl.h>

// ������������ ����� ����������� ���������� ��� ������ � ��������, ������� � ��������
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

// ������� ������ �� ����� (������������ ������� � ��������� ��������)
cl_float *srcA, *srcB, *dst;

cl_context cxGPUContext;        // OpenCL ��������
cl_command_queue cqCommandQueue;// OpenCL ������� �������
cl_platform_id cpPlatform;      // OpenCL ���������
cl_device_id cdDevice;          // OpenCL ����������
cl_program cpProgram;           // OpenCL ���������
cl_kernel ckKernel;             // OpenCL ����

cl_mem devSrcA;               // OpenCL ������ �� ���������� ��� ������� ������������� �������
cl_mem devSrcB;               // OpenCL ������ �� ���������� ��� ������� ������������� �������
cl_mem devDst;                // OpenCL ������ �� ���������� ��� ���������� �������� ��������

size_t globalWorkSize;        // ������ 1D ��������� ������������ ��������
size_t localWorkSize;		    // ������ 1D ������� ������
cl_int error;					// ���������� ��� ������ ���� ������

void Cleanup();
void CheckError(string errorPlace, cl_int err);
void CheckBuildError(cl_int err);
void PrintArray(cl_float *array, int count);
string getClCodeFromFile();

int main(int argc, char **argv) {

	// ������ ������������ ��������
	size_t countItems = 256;

	// ��������� ������� ����������� ������������ �������� (������ ��������) � ������� ������� ������ (1)
	globalWorkSize = countItems;
	localWorkSize = 1;

	// ��������� ������ �� ����� ��� ������������ �������� � ����������
	srcA = new cl_float[countItems];
	srcB = new cl_float[countItems];
	dst = new cl_float[countItems];

	// ������������� ������������ �������� ���������� ����������
	for (int i = 0; i < countItems; i++) {
		srcA[i] = i;
		srcB[i] = i * 2;
	}

	// ��������� OpenCL ���������
	CheckError("get platform", clGetPlatformIDs(1, &cpPlatform, NULL));

	// ��������� ���������
	CheckError("get devices", clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL));

	// �������� ���������
	cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &error);
	CheckError("create context", error);

	// �������� ������� �������
	cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &error);
	CheckError("create command queue", error);

	// ������ ���������� ����, ����������� ��� ���������� ������ �������
	size_t countBytesForVectors = sizeof(cl_float) * countItems;

	// ��������� ������ �� ���������� ��� ������� ������������� �������
	devSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, countBytesForVectors, NULL, &error);
	CheckError("create device buffer for A", error);

	// ��������� ������ �� ���������� ��� ������� ������������� �������
	devSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, countBytesForVectors, NULL, &error);
	CheckError("create device buffer for B", error);

	// ��������� ������ �� ���������� ��� ���������� �������� ��������
	devDst = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, countBytesForVectors, NULL, &error);
	CheckError("create device buffer for C", error);

	// ��������� ���� ���������� � ��������� ����������
	string codeStr = getClCodeFromFile();
	const char *openCLsrc = codeStr.c_str();

	// �������� ����������� �������� �� ������ ����� ����
	size_t kernelLength = 0;
	cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&openCLsrc, &kernelLength, &error);
	CheckError("create program", error);

	// ������ ������������ �� ����� ���� (��� �� �������� ���������� ��� ������
	// ���������� � ������� CheckBuildError() ����������� ����� ������������ ������)
	CheckBuildError(clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL));

	// �������� ����, ����������������� �� �������� ����� � ���������������� ���������
	ckKernel = clCreateKernel(cpProgram, "VectorAdd", &error);
	CheckError("create kernel", error);

	// ������������� ������ �� ������ ���������� � ���������� � �������������� ����������� ����
	CheckError("set kernel arg A", clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&devSrcA));
	CheckError("set kernel arg B", clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&devSrcB));
	CheckError("set kernel arg C", clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&devDst));
	CheckError("set kernel arg countElements", clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)&countItems));

	// ���������� � ������� ������� �� �������� ������ � ������ ���������� �� ������ �����
	// (CL_FALSE �������� ��� ���������� ������� �� ���� ���������� ����� ���� ������ ������� � �������)
	CheckError("write device A", clEnqueueWriteBuffer(cqCommandQueue, devSrcA, CL_FALSE, 0, countBytesForVectors, srcA, 0, NULL, NULL));
	CheckError("write device B", clEnqueueWriteBuffer(cqCommandQueue, devSrcB, CL_FALSE, 0, countBytesForVectors, srcB, 0, NULL, NULL));

	// ���������� � ������� �������� �� ������ ����
	CheckError("run kernel", clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL));

	// ���������� � ������� �������� �� �������� ���������� � ������ ����� �� ������ ����������
	// (CL_TRUE �������� ��� ���������� �������� ���� ���������� ���������� ���������� �������� � �������,
	// �. �. ���������� ����. ��� ����������� ������������ ������ �� ������ �� �������� �� ����)
	CheckError("receive result", clEnqueueReadBuffer(cqCommandQueue, devDst, CL_TRUE, 0, countBytesForVectors, dst, 0, NULL, NULL));

	// ����� �� ������� ������� ������������� �������
	cout << "Src A: " << endl;
	PrintArray(srcA, countItems);
	cout << endl;

	// ����� �� ������� ������� ������������� �������
	cout << "Src B: " << endl;
	PrintArray(srcB, countItems);
	cout << endl;

	// ����� �� ������� ���������� �������� ��������
	cout << "Result: " << endl;
	PrintArray(dst, countItems);
	cout << endl;

	cin.get();

	// ������� ������
	Cleanup();
}

// ������� ������� ������. ������� �������� OpenCL ��������������
// ������������ ��������� �� ���������� OpenCL
void Cleanup() {
	if (ckKernel)
		clReleaseKernel(ckKernel);

	if (cpProgram)
		clReleaseProgram(cpProgram);

	if (cqCommandQueue)
		clReleaseCommandQueue(cqCommandQueue);

	if (cxGPUContext)
		clReleaseContext(cxGPUContext);

	if (devSrcA)
		clReleaseMemObject(devSrcA);

	if (devSrcB)
		clReleaseMemObject(devSrcB);

	if (devDst)
		clReleaseMemObject(devDst);

	// ������� ������ ��� ������� �� ������ �����
	delete srcA;
	delete srcB;
	delete dst;
}

// ������� ��������� ������, ������� ����� ����
// �������� �� ��������� ������� OpenCL
void CheckError(string errorPlace, cl_int err) {
	if (err == CL_SUCCESS) {
		return;
	}

	cout << "Error in " << errorPlace << " (code: " << err << ")" << endl;
	cin.get();
	Cleanup();
	exit(0);
}

// ������� ��������� ������, ������� ����� ����
// �������� �� ������� ������ OpenCL ����
void CheckBuildError(cl_int err) {
	if (err == CL_SUCCESS) {
		return;
	}

	// ��������� ���� ������ ��������� (����� ���������� ������ ����������)
	char buildLog[50000];
	clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
	cout << "Build log:" << endl << buildLog << endl;

	CheckError("build program", err);
}

// ������� ������ �������� ��������� ������� �� �������
void PrintArray(cl_float *array, int count) {
	for (int i = 0; i < count; i++) {
		cout << array[i] << "; ";
	}
	cout << endl;
}

// ��������� ���� ���������� �� ����� � ��������� ����������
string getClCodeFromFile() {
	ifstream ifs("device.cl");
	string codeStr(
		(istreambuf_iterator<char>(ifs)),
		(istreambuf_iterator<char>()));

	cout << codeStr << endl;
	return codeStr;
}
