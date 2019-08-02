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



                                 /**
                                  * Declare the size of each submatrix (it must be
                                  * the same work-group size declared in the host code).
                                  */

                                 const int SUB_SIZE = 8;

                                 /**
                                  * Get work-item identifiers.
                                  */
                                 int colIndex = get_local_id(0);
                                 int rowIndex = get_local_id(1);
                                 int globalColIndex = get_global_id(0);
                                 int globalRowIndex = get_global_id(1);
                                 int index = (globalRowIndex * imgWidth) + globalColIndex;

                                 //lida com bordas

                             if(globalColIndex < maskSize/2
                             || globalRowIndex < maskSize/2
                             || globalColIndex >= imgWidth - maskSize/2
                             || globalRowIndex >= imgHeight - maskSize/2){
                                 outputImg[index] = 0 ;

                                 return;
                             }
                             /**
                              * Create submatrices that will cache the matrices A and B in local memory.
                              */

                             __local int aSub[SUB_SIZE][SUB_SIZE];

                             /**
                              * Initialize accumulator register.
                              */
                             int sum = 0;

                             /**
                              * Loop over all submatrices.
                              */

                             for(int s = 0; s < nSub; s++){

                                 /**
                                  * Load submatrices into local memory.
                                  */

                                 const int sCol = SUB_SIZE * s + colIndex;
                                 const int sRow = SUB_SIZE * s + rowIndex;
                                 aSub[rowIndex][colIndex] = a[globalRowIndex * K + sCol];


                                 /**
                                  * Synchronize all work-items in this work-group.
                                  */

                                 barrier(CLK_LOCAL_MEM_FENCE);

                                 /**
                                  * Perform the computation for a single submatrix.
                                  */


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
