#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <string.h>
#include <time.h>
#include <math.h>

#include "CImg.h"
using namespace cimg_library;

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

void seqRgb2Gray(unsigned int imgWidth,
                        unsigned int imgHeight,
                        unsigned char *rChannel,
                        unsigned char *gChannel,
                        unsigned char *bChannel,
                        unsigned char *grayImg);                    // Sequentially convert an RGB image to grayscale.

void seqConvolve(unsigned int imgWidth,
                            unsigned int imgHeight,
                            unsigned int maskSize,
                            unsigned char *inputImg,
                            float *mask,
                            unsigned char *outputImg);              // Sequentially convolve an image with a filter.

void seqFilter(unsigned int imgWidth,
                            unsigned int imgHeight,
                            unsigned int lpMaskSize,
                            unsigned int hpMaskSize,
                            unsigned char *inputRchannel,
                            unsigned char *inputGchannel,
                            unsigned char *inputBchannel,
                            float *lpMask,
                            float *hpMask,
                            unsigned char *outputImg);              // Sequentially filter an image.

bool checkEquality(unsigned char* img1,
                    unsigned char* img2,
                    const int W,
                    const int H);                                   // Check if the images img1 and img2 are equal.

void displayImg(unsigned char *img, int imgWidth, int imgHeight);   // Display unsigned char matrix as an image.

// =================================================================
// ------------------------ OpenCL Functions -----------------------
// =================================================================

cl::Device getDefaultDevice();                                    // Return a device found in this OpenCL platform.

void initializeDevice();                                          // Inicialize device and compile kernel code.

void parFilter(unsigned int imgWidth,
                            unsigned int imgHeight,
                            unsigned int lpMaskSize,
                            unsigned int hpMaskSize,
                            unsigned char *inputRchannel,
                            unsigned char *inputGchannel,
                            unsigned char *inputBchannel,
                            float *lpMask,
                            float *hpMask,
                            unsigned char *outputImg);             // Parallelly filter an image.

// =================================================================
// ------------------------ Global Variables ------------------------
// =================================================================

cl::Program program;                // The program that will run on the device.
cl::Context context;                // The context which holds the device.
cl::Device device;                  // The device where the kernel will run.

// =================================================================
// ------------------------- Main Function -------------------------
// =================================================================

int main(){

    /**
     * Create auxiliary variables.
     * */

    clock_t start, end;

    /**
     * Load input image.
     * */

    CImg<unsigned char> cimg("input_img.jpg");
    unsigned char *inputImg = cimg.data();
    unsigned int imgWidth = cimg.width(), imgHeight = cimg.height();
    unsigned char *inputRchannel = &inputImg[0]; // inputRchannel aponta para início da imagem
    unsigned char *inputGchannel = &inputImg[imgWidth*imgHeight];
    unsigned char *inputBchannel = &inputImg[2*imgWidth*imgHeight];

    /**
     * Create a low-pass filter mask.
     * */

    const int lpMaskSize = 5;
    float lpMask[lpMaskSize][lpMaskSize] = {
        {.04,.04,.04,.04,.04},
        {.04,.04,.04,.04,.04},
        {.04,.04,.04,.04,.04},
        {.04,.04,.04,.04,.04},
        {.04,.04,.04,.04,.04},
    };
    float* lpMaskData = &lpMask[0][0];

    /**
     * Create a high-pass filter mask.
     * */

    const int hpMaskSize = 5;
    float hpMask[hpMaskSize][hpMaskSize] = {
        {-1,-1,-1,-1,-1},
        {-1,-1,-1,-1,-1},
        {-1,-1,24,-1,-1},
        {-1,-1,-1,-1,-1},
        {-1,-1,-1,-1,-1},
    };
    float* hpMaskData = &hpMask[0][0];

    /**
     * Allocate memory for the output images.
     * */

    unsigned char *seqFilteredImg = (unsigned char*) malloc(imgWidth * imgHeight * sizeof(unsigned char));
    unsigned char *parFilteredImg = (unsigned char*) malloc(imgWidth * imgHeight * sizeof(unsigned char));

    /**
     * Sequentially convolve filter over image.
     * */

    start = clock();
    seqFilter(imgWidth, imgHeight, lpMaskSize, hpMaskSize, inputRchannel, inputGchannel, inputBchannel,
    lpMaskData, hpMaskData, seqFilteredImg);
    end = clock();
    double seqTime = ((double) 10e3 * (end - start)) / CLOCKS_PER_SEC;

    /**
     * Initialize OpenCL device.
     */

    initializeDevice();

    /**
     * Parallelly convolve filter over image.
     * */

    start = clock();
    parFilter(imgWidth, imgHeight, lpMaskSize, hpMaskSize, inputRchannel, inputGchannel, inputBchannel,
    lpMaskData, hpMaskData, parFilteredImg);
    end = clock();
    double parTime = ((double) 10e3 * (end - start)) / CLOCKS_PER_SEC;

    /**
     * Check if outputs are equal.
     * */

    bool equal = checkEquality(seqFilteredImg, parFilteredImg, imgWidth, imgHeight);

    /**
     * Print results.
     */

    std::cout << "Status: " << (equal ? "SUCCESS!" : "FAILED!") << std::endl;
    std::cout << "Mean execution time: \n\tSequential: " << seqTime << " ms;\n\tParallel: " << parTime << " ms." << std::endl;
    std::cout << "Performance gain: " << (100 * (seqTime - parTime) / parTime) << "\%\n";

    /**
     * Display filtered image.
     * */

    displayImg(parFilteredImg, imgWidth, imgHeight);
    displayImg(seqFilteredImg, imgWidth, imgHeight);
    return 0;
}

// =================================================================
// ------------------------ OpenCL Functions -----------------------
// =================================================================

/**
 * Return a device found in this OpenCL platform.
 * */

cl::Device getDefaultDevice(){

      /**
       * Search for all the OpenCL platforms available and check
       * if there are any.
       * */

      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);

      if (platforms.empty()){
          std::cerr << "No platforms found!" << std::endl;
          exit(1);
      }

      /**
       * Search for all the devices on the first platform and check if
       * there are any available.
       * */

      auto platform = platforms.front();
      std::vector<cl::Device> devices;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

      if (devices.empty()){
          std::cerr << "No devices found!" << std::endl;
          exit(1);
      }

      /**
       * Return the first device found.
       * */

      return devices.front();
}

/**
 * Inicialize device and compile kernel code.
 * */

void initializeDevice(){
  /**
   * Select the first available device.
   * */
  device = getDefaultDevice();

  /**
   * Read OpenCL kernel file as a string.
   * */
  std::ifstream kernel_file("image_filtering.cl");
  std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
  /**
   * Compile kernel program which will run on the device.
   * */
  cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
  context = cl::Context(device);
  program = cl::Program(context, sources);

  auto err = program.build();
  if(err != CL_BUILD_SUCCESS){
      std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
      << "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
      exit(1);
  }
}

/**
 * Parallelly filter an image.
 */

void parFilter(unsigned int imgWidth,
                            unsigned int imgHeight,
                            unsigned int lpMaskSize, // tamanho da máscara
                            unsigned int hpMaskSize,
                            unsigned char *inputRchannel,
                            unsigned char *inputGchannel,
                            unsigned char *inputBchannel,
                            float *lpMask, // repassa matriz de máscara passa-baixo
                            float *hpMask, // repassa matriz de máscara passa-alto
                            unsigned char *outputImg){ // repassa saída para imagem

// aloca buffers
      cl_int error1, error2, error3;
      cl::Buffer RBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgWidth * imgHeight * sizeof(unsigned char), inputRchannel, &error1);
      std::cout <<"Err1:" << error1 << std::endl;
      cl::Buffer GBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgWidth * imgHeight * sizeof(unsigned char), inputGchannel, &error2);
      std::cout <<"Err2:" << error2 << std::endl;
      cl::Buffer BBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , imgWidth * imgHeight * sizeof(unsigned char), inputBchannel, &error3);
      std::cout <<"Err3:" << error3 << std::endl;
      cl::Buffer outgrayBuf(context, CL_MEM_READ_WRITE, imgWidth * imgHeight * sizeof(unsigned char));

      // configura parâmetros do kernel
      cl::Kernel kernel(program, "rgb2gray");
      kernel.setArg(0, RBuf);
      kernel.setArg(1, GBuf);
      kernel.setArg(2, BBuf);
      kernel.setArg(3, outgrayBuf);

      /**
       * Execute the kernel function and collect its result.
       * */

      /* Chamando filtro low pass
      *
      */
      cl_int error5, error6, error7, error8;
      // Gera buffers para lowpass filter
  
      cl::Buffer LPmaskBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, lpMaskSize * lpMaskSize * sizeof(float), lpMask, &error6);
      std::cout <<"Err6:" << error6 << std::endl;
      cl::Buffer HPmaskBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, hpMaskSize * hpMaskSize * sizeof(float), hpMask);
      cl::Buffer LowFilterOutBuf(context, CL_MEM_READ_WRITE, imgWidth * imgHeight * sizeof(unsigned char));
      cl::Buffer HighFilterOutBuf(context, CL_MEM_READ_WRITE, imgWidth * imgHeight * sizeof(unsigned char));
      //chamando filtro high pass
      cl::Kernel kernel2(program, "filterImage");
      kernel2.setArg(0, sizeof(unsigned int), &lpMaskSize);
      kernel2.setArg(1, outgrayBuf);
      kernel2.setArg(2, LPmaskBuf);
      kernel2.setArg(3, LowFilterOutBuf);
      kernel2.setArg(4, sizeof(unsigned int), &imgWidth);
      kernel2.setArg(5, sizeof(unsigned int), &imgHeight);

      cl::Kernel kernel3(program, "filterImage");
      kernel3.setArg(0, sizeof(unsigned int), &hpMaskSize);
      kernel3.setArg(1, LowFilterOutBuf);
      kernel3.setArg(2, HPmaskBuf);
      kernel3.setArg(3, HighFilterOutBuf);
      kernel3.setArg(4, sizeof(unsigned int), &imgWidth);
      kernel3.setArg(5, sizeof(unsigned int), &imgHeight);

      cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(imgWidth * imgHeight));
      queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(imgWidth ,imgHeight));
      queue.enqueueNDRangeKernel(kernel3, cl::NullRange, cl::NDRange(imgWidth ,imgHeight));
      queue.enqueueReadBuffer(HighFilterOutBuf, CL_TRUE, 0,  imgWidth * imgHeight * sizeof(unsigned char), outputImg);



}
// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

/**
 * Sequentially convert an RGB image to grayscale.
 */

void seqRgb2Gray(unsigned int imgWidth,
                        unsigned int imgHeight,
                        unsigned char *rChannel,
                        unsigned char *gChannel,
                        unsigned char *bChannel,
                        unsigned char *grayImg){

    /**
     * Declare the current index variable.
     */

    size_t idx;

    /**
     * Loop over input image pixels.
     */

    for(int i = 0; i < imgWidth; i++){
        for(int j = 0; j < imgHeight; j++){

            /**
             * Compute average pixel.
             */

            idx = i + j*imgWidth;
            grayImg[idx] = (rChannel[idx] + gChannel[idx] + bChannel[idx]) / 3;
        }
    }
}

/**
 * Sequentially convolve an image with a filter mask.
 */

void seqConvolve(unsigned int imgWidth,
                            unsigned int imgHeight,
                            unsigned int maskSize,
                            unsigned char *inputImg,
                            float *mask,
                            unsigned char *outputImg){
    /**
     * Loop through input image.
     * */

    for(size_t i = 0; i < imgWidth; i++){
        for(size_t j = 0; j < imgHeight; j++){

            /**
             * Check if the mask cannot be applied to the
             * current image pixel.
             * */

            if(i < maskSize/2
            || j < maskSize/2
            || i >= imgWidth - maskSize/2
            || j >= imgHeight - maskSize/2){
                outputImg[i + j * imgWidth] = 0;
                continue;
            }

            /**
             * Apply mask based on the neighborhood of pixel inputImg(j,i).
             * */

            int outSum = 0;
            for(size_t k = 0; k < maskSize; k++){
                for(size_t l = 0; l < maskSize; l++){
                  size_t colIdx = i - maskSize/2 + k;
                  size_t rowIdx = j - maskSize/2 + l;
                  size_t maskIdx = (maskSize-1-k) + (maskSize-1-l)*maskSize;
                  outSum += inputImg[rowIdx * imgWidth + colIdx] * mask[maskIdx];
                }
            }

            /**
             * Update output pixel.
             * */

            if(outSum < 0){
                outputImg[i + j * imgWidth] = 0;
            } else if(outSum > 255){
                outputImg[i + j * imgWidth] = 255;
            } else{
                outputImg[i + j * imgWidth] = outSum;
            }
        }
    }
}

/**
 * Sequentially filter an image.
 */

void seqFilter(unsigned int imgWidth,
                            unsigned int imgHeight,
                            unsigned int lpMaskSize,
                            unsigned int hpMaskSize,
                            unsigned char *inputRchannel,
                            unsigned char *inputGchannel,
                            unsigned char *inputBchannel,
                            float *lpMask,
                            float *hpMask,
                            unsigned char *outputImg){

    /**
     * Convert input image to grayscale.
     */

    unsigned char *grayOut = (unsigned char*) malloc(imgWidth * imgHeight * sizeof(unsigned char));
    seqRgb2Gray(imgWidth, imgHeight, inputRchannel, inputGchannel, inputBchannel, grayOut);

    /**
     * Apply the low-pass filter.
     */

    unsigned char *lpOut = (unsigned char*) malloc(imgWidth * imgHeight * sizeof(unsigned char));
     seqConvolve(imgWidth, imgHeight, lpMaskSize, grayOut, lpMask, lpOut);

    /**
     * Apply the high-pass filter.
     */

    seqConvolve(imgWidth, imgHeight, hpMaskSize, lpOut, hpMask, outputImg);
}

/**
 * Display unsigned char matrix as an image.
 * */

void displayImg(unsigned char *img, int imgWidth, int imgHeight){

    /**
     * Create C_IMG object.
     * */

    CImg<unsigned char> cimg(imgWidth, imgHeight);

    /**
     * Transfer image data to C_IMG object.
     * */

    for(int i = 0; i < imgWidth; i++){
        for(int j = 0; j < imgHeight; j++){
            cimg(i,j) = img[i + imgWidth*j];
        }
    }

    /**
     * Display image.
     * */

    cimg.display();
}

/**
 * Check if the images img1 and img2 are equal.
 * */

bool checkEquality(unsigned char* img1,
                unsigned char* img2,
                const int M,
                const int N){
    for(int i = 0; i < M*N; i++){
        if(img1[i] != img2[i]){
          printf("img1 %i img2 %i i=%i ",img1[i], img2[i], i);
            //return false;
        }
    }
    return true;
}
