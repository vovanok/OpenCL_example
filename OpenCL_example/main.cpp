// Основной заголовочный файл OpenCL
#include <CL/cl.h>

// Заголовочные файлы стандартной библиотеки для работы с консолью, файлами и строками
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

// Массивы данных на хосте (складываемые вектора и результат сложения)
cl_float *srcA, *srcB, *dst;

cl_context cxGPUContext;        // OpenCL контекст
cl_command_queue cqCommandQueue;// OpenCL очередь комманд
cl_platform_id cpPlatform;      // OpenCL платформа
cl_device_id cdDevice;          // OpenCL устройство
cl_program cpProgram;           // OpenCL программа
cl_kernel ckKernel;             // OpenCL ядро

cl_mem devSrcA;					// OpenCL память на устройстве для первого складываемого вектора
cl_mem devSrcB;					// OpenCL память на устройстве для второго складываемого вектора
cl_mem devDst;					// OpenCL память на устройстве для результата сложения векторов

size_t globalWorkSize;			// Размер 1D глобаного пространства индексов
size_t localWorkSize;			// Размер 1D рабочей группы
cl_int error;					// Переменная для записи кода ошибки

void Cleanup();
void CheckError(string errorPlace, cl_int err);
void CheckBuildError(cl_int err);
void PrintArray(cl_float *array, int count);
string getClCodeFromFile();

int main(int argc, char **argv) {

	// Размер складываемых векторов
	size_t countItems = 256;

	// Установка размера глобального пространства индексов (размер векторов) и размера рабочей группы (1)
	globalWorkSize = countItems;
	localWorkSize = 1;

	// Выделение памяти на хосте для складываемых векторов и результата
	srcA = new cl_float[countItems];
	srcB = new cl_float[countItems];
	dst = new cl_float[countItems];

	// Инициализация складываемых векторов начальными значениями
	for (int i = 0; i < countItems; i++) {
		srcA[i] = i;
		srcB[i] = i * 2;
	}

	// Получений OpenCL платформы
	CheckError("get platform", clGetPlatformIDs(1, &cpPlatform, NULL));

	// Получение устройств
	CheckError("get devices", clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL));

	// Создание контекста
	cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &error);
	CheckError("create context", error);

	// Создание очереди комманд
	cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &error);
	CheckError("create command queue", error);

	// Расчет количества байт, необходимых для размещения одного вектора
	size_t countBytesForVectors = sizeof(cl_float) * countItems;

	// Выделение памяти на устройстве для первого складываемого вектора
	devSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, countBytesForVectors, NULL, &error);
	CheckError("create device buffer for A", error);

	// Выделение памяти на устройстве для второго складываемого вектора
	devSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, countBytesForVectors, NULL, &error);
	CheckError("create device buffer for B", error);

	// Выделение памяти на устройстве для результата сложения векторов
	devDst = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, countBytesForVectors, NULL, &error);
	CheckError("create device buffer for C", error);

	// Получение кода устройства в строковую переменную
	string codeStr = getClCodeFromFile();
	const char *openCLsrc = codeStr.c_str();

	// Создание программной сущности на основе строк кода
	size_t kernelLength = 0;
	cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&openCLsrc, &kernelLength, &error);
	CheckError("create program", error);

	// Сборка загруженного из файла кода (при не успешном выполнении код ошибки
	// передается в функцию CheckBuildError() выполняющую вывод подробностей ошибки)
	CheckBuildError(clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL));

	// Создание ядра, инициализируемого по строкому имени в скомпилированной программе
	ckKernel = clCreateKernel(cpProgram, "VectorAdd", &error);
	CheckError("create kernel", error);

	// Сопоставление ссылок на память устройства и переменных с соответвующими параметрами ядра
	CheckError("set kernel arg A", clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&devSrcA));
	CheckError("set kernel arg B", clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&devSrcB));
	CheckError("set kernel arg C", clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&devDst));
	CheckError("set kernel arg countElements", clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)&countItems));

	// Постановка в очередь комманд на передачу данных в память устройства из памяти хоста
	// (CL_FALSE означает что выполнение комманд не ждет завершения каких либо других комманд в очереди)
	CheckError("write device A", clEnqueueWriteBuffer(cqCommandQueue, devSrcA, CL_FALSE, 0, countBytesForVectors, srcA, 0, NULL, NULL));
	CheckError("write device B", clEnqueueWriteBuffer(cqCommandQueue, devSrcB, CL_FALSE, 0, countBytesForVectors, srcB, 0, NULL, NULL));

	// Постановка в очередь комманды на запуск ядра
	CheckError("run kernel", clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL));

	// Постановка в очередь комманды на передачу результата в память хоста из памяти устройства
	// (CL_TRUE означает что выполнение комманды ждет завершения выполнения предыдущей комманды в очереди,
	// т. е. выполнения ядра. Это гарантирует актуальность данных на момент их передачи на хост)
	CheckError("receive result", clEnqueueReadBuffer(cqCommandQueue, devDst, CL_TRUE, 0, countBytesForVectors, dst, 0, NULL, NULL));

	// Вывод на консоль первого складываемого вектора
	cout << "Src A: " << endl;
	PrintArray(srcA, countItems);
	cout << endl;

	// Вывод на консоль второго складываемого вектора
	cout << "Src B: " << endl;
	PrintArray(srcB, countItems);
	cout << endl;

	// Вывод на консоль результата сложения векторов
	cout << "Result: " << endl;
	PrintArray(dst, countItems);
	cout << endl;

	cin.get();

	// Очистка данных
	Cleanup();
}

// Функция очистки данных. Очистка объектов OpenCL осуществляются
// специальными функциями из библиотеки OpenCL
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

	// Очистка памяти под вектора из памяти хоста
	delete srcA;
	delete srcB;
	delete dst;
}

// Функция обработки ошибки, которая может быть
// получена из различных функций OpenCL
void CheckError(string errorPlace, cl_int err) {
	if (err == CL_SUCCESS) {
		return;
	}

	cout << "Error in " << errorPlace << " (code: " << err << ")" << endl;
	cin.get();
	Cleanup();
	exit(0);
}

// Функция обработки ошибки, которая может быть
// получена из функции сборки OpenCL кода
void CheckBuildError(cl_int err) {
	if (err == CL_SUCCESS) {
		return;
	}

	// Получение лога сборки программы (здесь содержатся ошибки компиляции)
	char buildLog[50000];
	clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
	cout << "Build log:" << endl << buildLog << endl;

	CheckError("build program", err);
}

// Функция вывода значений элементов массива на консоль
void PrintArray(cl_float *array, int count) {
	for (int i = 0; i < count; i++) {
		cout << array[i] << "; ";
	}
	cout << endl;
}

// Получение кода устройства из файла в строковую переменную
string getClCodeFromFile() {
	ifstream ifs("device.cl");
	string codeStr(
		(istreambuf_iterator<char>(ifs)),
		(istreambuf_iterator<char>()));

	cout << codeStr << endl;
	return codeStr;
}
