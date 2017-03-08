/*
 * Sobel filter on pgm image with OpenMPI
 * Author: CHANG GAO
 * Development platform: g++ (Ubuntu 5.4.1-2ubuntu1~14.04) 5.4.1 20160904
 * Last modified date: 10 Feb 2017
 * Compilation: mpic++ -std=c++11 Sobel.cpp -o Sobel
                mpirun -np <num_of_process> ./Sobel <input_image> <output_image>
 */

#include "mpi.h"
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>

// ***************** Add/Change the functions(including processImage) here ********************* 

int* processImage(int* imageInfo, int* chunkInfo){
    int sum, sumx, sumy;
    int GX[3][3], GY[3][3];
    /* 3x3 Sobel masks. */
    GX[0][0] = -1; GX[0][1] =  0; GX[0][2] =  1;
    GX[1][0] = -2; GX[1][1] =  0; GX[1][2] =  2;
    GX[2][0] = -1; GX[2][1] =  0; GX[2][2] =  1;
    GY[0][0] =  1; GY[0][1] =  2; GY[0][2] =  1;
    GY[1][0] =  0; GY[1][1] =  0; GY[1][2] =  0;
    GY[2][0] = -1; GY[2][1] = -2; GY[2][2] = -1;

    int image_height = chunkInfo[0]-2, image_width = chunkInfo[1];
    int* outputChunk = new int[image_height*image_width];
    std::fill(outputChunk, outputChunk + image_height*image_width, 0);
	for(int x = 1; x <= image_height; x++){
		for(int y = 1; y < image_width-1; y++){
			sumx = sumy = 0;
		    for(int i=-1; i<=1; i++){
				for(int j=-1; j<=1; j++){
					sumx += imageInfo[(x+i)*image_width+(y+j)] * GX[i+1][j+1];
                    sumy += imageInfo[(x+i)*image_width+(y+j)] * GY[i+1][j+1];
				}
			}
			sum = abs(sumx) + abs(sumy);
			outputChunk[(x-1)*image_width + y] = sum < 0 ? 0 : sum > 255 ? 255 : sum;
		}
	}
	return outputChunk;
}

int main(int argc, char* argv[]){
	int processId, num_processes, image_height, image_width, image_maxShades;
	int *inputImage;
	
	// Setup MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	
    if(argc != 3){
		if(processId == 0)
			std::cout << "ERROR: Incorrect number of arguments. Format is: <Input image filename> <Output image filename>" << std::endl;
		MPI_Finalize();
        return 0;
    }
	
	if(processId == 0){
		std::ifstream file(argv[1]);
		if(!file.is_open()){
			std::cout << "ERROR: Could not open file " << argv[1] << std::endl;
			MPI_Finalize();
			return 0;
		}

		std::cout << "Detect edges in " << argv[1] << " using " << num_processes << " processes" << std::endl;

		std::string workString;
		/* Remove comments '#' and check image format */ 
		while(std::getline(file,workString)){
			if( workString.at(0) != '#' ){
				if( workString.at(1) != '2' ){
					std::cout << "Input image is not a valid PGM image" << std::endl;
					return 0;
				} else {
					break;
				}       
			}
		}
		/* Check image size */ 
		while(std::getline(file,workString)){
			if(workString.at(0) != '#'){
				std::stringstream stream(workString);
				int n;
				stream >> n;
				image_width = n;
				stream >> n;
				image_height = n;
				break;
			}
		}

		/* Check image max shades */ 
		while(std::getline(file,workString)){
			if (workString.at(0) != '#'){
				std::stringstream stream(workString);
				stream >> image_maxShades;
				break;
			}
		}

		inputImage = new int[image_height*image_width];

		/* Fill input image matrix */ 
		int pixel_val;
		for(int i = 0; i < image_height; i++){
			if(std::getline(file,workString) && workString.at(0) != '#'){
				std::stringstream stream(workString);
				for(int j = 0; j < image_width; j++){
					if(!stream) break;
					stream >> pixel_val;
					inputImage[i*image_width+j] = pixel_val;
				}
			}
		}
	} // Done with reading image using process 0
	
	// ***************** Add code as per your requirement below ********************* 

    int *inputChunk = NULL, *outputImage = NULL, *imageInfo = NULL, *newInputImage = NULL;
    int *chunkInfo = new int[2], *outputChunk = NULL;

    if (processId == 0) {
    	int rows = ceil(image_height*1.0/num_processes);
    	inputChunk = new int[num_processes*2];
        for (int i = 0; i < num_processes; i++) {
        	inputChunk[2*i]   = rows+2;
        	inputChunk[2*i+1] = image_width;
        }
        outputImage = new int[rows*num_processes*image_width];
        // modify inputImage so that the last/first line in a imagechunk will be kept
        newInputImage = new int[(rows+2)*num_processes*image_width];
        std::fill(newInputImage, newInputImage + (rows+2)*num_processes*image_width, 0);
        for (int p = 0; p < num_processes; ++p) {
        	// set first row from previous chunk
        	if (p*rows > image_height) break;
        	if (p) {
        		for (int j = 0; j < image_width; ++j){
        			newInputImage[p*(rows+2)*image_width + j] = inputImage[(p*rows-1)*image_width+j];
        		}
        	}
        	// copy rows for each chunk
        	for (int i = 0; i < rows; ++i){
        		if (p*rows+i+1 > image_height) break;
        		for (int j = 0; j < image_width; ++j){
        			newInputImage[(p*(rows+2)+i+1)*image_width + j] = inputImage[(p*rows+i)*image_width+j];
        		}
        	}
        	// set last row from next chunk
        	if ((p+1)*rows+1 > image_height) break;
        	if (p != num_processes - 1) {
        		for (int j = 0; j < image_width; ++j) {
        			newInputImage[((p+1)*(rows+2)-1)*image_width + j] = inputImage[(p+1)*rows*image_width+j];
        		}
        	}
        }
    }

    // scatter image info 
    MPI_Scatter(
    	inputChunk, // void* send_data
        2, // int send_count
        MPI_INT, // MPI_Datatype send_datatype
        chunkInfo, // void* recv_data
        2, // int recv_count
        MPI_INT, // MPI_Datatype recv_datatype
        0, // int root
        MPI_COMM_WORLD  // MPI_Comm communicator
    );
    std::cout << "Process " << processId << " finished scattering input image info.\n";

    imageInfo = new int[chunkInfo[0]*chunkInfo[1]];
    std::fill(imageInfo, imageInfo + chunkInfo[0]*chunkInfo[1], 0);

    // scatter image chunk 
    MPI_Scatter(
    	newInputImage, // void* send_data
        chunkInfo[0]*chunkInfo[1], // int send_count
        MPI_INT, // MPI_Datatype send_datatype
        imageInfo, // void* recv_data
        chunkInfo[0]*chunkInfo[1], // int recv_count
        MPI_INT, // MPI_Datatype recv_datatype
        0, // int root
        MPI_COMM_WORLD // MPI_Comm communicator
    );
    std::cout << "Process " << processId << " finished scattering input image chunk.\n";
    
    outputChunk = processImage(imageInfo, chunkInfo);
    std::cout << "Process " << processId << " finished calculation.\n";

    // gather image chunk
    MPI_Gather(
    	outputChunk, // void* send_data
        (chunkInfo[0]-2)*chunkInfo[1], // int send_count
        MPI_INT, // MPI_Datatype send_datatype
        outputImage, // void* recv_data
        (chunkInfo[0]-2)*chunkInfo[1], // int recv_count
        MPI_INT, // MPI_Datatype recv_datatype
        0, // int root
        MPI_COMM_WORLD // MPI_Comm communicator
    );
    std::cout << "Process " << processId << " finished gathering output image chunk.\n";
    
	if (processId == 0) {
		// Start writing output to your file
		std::ofstream ofile(argv[2]);
		if (ofile.is_open()) {
			ofile << "P2" << "\n" << image_width << " " << image_height << "\n" << image_maxShades << "\n";
			// first row
			for (int j = 0; j < image_width; j++) ofile << 0 << " ";
			ofile << "\n";
			for (int i = 1; i < image_height -1; i++) {
				for (int j = 0; j < image_width; j++) {
					ofile << outputImage[i*image_width+j] << " ";
				}
				ofile << "\n";
			}
			// last row
			for (int j = 0; j < image_width; j++) ofile << 0 << " ";
			ofile << "\n";
			delete [] inputChunk, outputImage, inputImage, newInputImage;
		} else {
			std::cout << "ERROR: Could not open output file " << argv[2] << std::endl;
			return 0;
		}	
	}
	delete [] chunkInfo, imageInfo, outputChunk;

    MPI_Finalize();
    return 0;
}