__kernel void rgb2gray(__global unsigned char* inputRchannel,
                       __global unsigned char* inputGchannel,
                       __global unsigned char* inputBchannel,
                       __global unsigned char* outputImg){



                      const int index = get_global_id(0);
                      outputImg[index] = (inputRchannel[index] + inputGchannel[index] + inputBchannel[index])/3;

                       }

__kernel void filterImage( unsigned int maskSize,
                            __global unsigned char* inputImg,
                            __constant float* mask,
                            __global unsigned char* outputImg,
                            unsigned int imgWidth,
                            unsigned int imgHeight ){

                            int colIndex = get_global_id(0);
                            int rowIndex = get_global_id(1);

                            if(colIndex < maskSize/2
                            || rowIndex < maskSize/2
                            || colIndex >= imgWidth - maskSize/2
                            || rowIndex >= imgHeight - maskSize/2){
                                outputImg[colIndex + rowIndex * imgWidth] = 0 ;
                                return;
                            }


                            /**
                             * Initialize accumulator register.
                             */
                            int sum = 0;


                            for(int i = 0 ; i < maskSize; i++){
                                for(int j = 0; j < maskSize; j++){

                                  int colIdx = colIndex - maskSize/2 + i;
                                  int rowIdx = rowIndex - maskSize/2 + j;
                                  int maskIdx = (maskSize-1-i) + (maskSize-1-j)*maskSize;
                                  sum += inputImg[rowIdx * imgWidth + colIdx] * mask[maskIdx];
                                }

                            }

                            //Atualizando valores dos pixels de saída
                          if(sum < 0){
                              outputImg[rowIndex * imgWidth + colIndex] = 0;
                          } else if(sum > 255){
                              outputImg[rowIndex * imgWidth + colIndex] = 255;
                          }else{
                              outputImg[rowIndex * imgWidth + colIndex] = sum;
                          }


}
