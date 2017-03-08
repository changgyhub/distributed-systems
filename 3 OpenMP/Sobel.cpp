/*
 * Author: CHANG GAO
 * Development platform: g++ (Ubuntu 5.4.1-2ubuntu1~14.04) 5.4.1 20160904
 * Last modified date: 6 March 2017
 * Compilation: g++ -fopenmp Implementation.cpp -o Sobel
                export OMP_NUM_THREADS=<#threads>
                ./Sobel <Input image filename> <Output image filename> <Chunk size> <a1/a2>
 * Test platform: openlab.ics.uci.edu
 */

#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
 
/* Global variables, Look at their usage in main() */
int image_height;
int image_width;
int image_maxShades;
int inputImage[1000][1000];
int outputImage[1000][1000];
int chunkSize;
int maskX[3][3];
int maskY[3][3];
std::vector<std::pair<int, int> > thread_rows;

/* ****************Change and add functions below ***************** */

void Sobel(int chunkcnt){
    int sumx, sumy, sum;
    for(int x = chunkSize*chunkcnt; x < chunkSize*(chunkcnt+1); ++x){
        for(int y = 0; y < image_width; ++y){
            sumx = 0; sumy = 0;
            if( x <= 0 || x >= (image_height-1) || y <= 0 || y >= (image_width-1)) sum = 0;
            else {
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j){
                        sumx += inputImage[x+i][y+j] * maskX[i+1][j+1];
                        sumy += inputImage[x+i][y+j] * maskY[i+1][j+1];
                    }
                }
                sum = (abs(sumx) + abs(sumy));
            }
            outputImage[x][y] = sum < 0 ? 0 : sum > 255 ? 255 : sum;
        }
    }
}

void compute_sobel_static() {
    int thread_id, num_chunks = ceil(image_height*1.0/chunkSize);

// start static scheduling
#pragma omp parallel for schedule(static) private(thread_id) 
    for (int i = 0; i < num_chunks; ++i){
        thread_id = omp_get_thread_num();

// std::vector.push_back() should be an opt in critical section
#pragma omp critical
        thread_rows.push_back(std::make_pair(thread_id, i*chunkSize));

        Sobel(i);
    }
}

void compute_sobel_dynamic() {
    int thread_id, num_chunks = ceil(image_height*1.0/chunkSize);

// start dynamic scheduling
#pragma omp parallel for schedule(dynamic) private(thread_id) 
    for (int i = 0; i < num_chunks; ++i){
        thread_id = omp_get_thread_num();

// std::vector.push_back() should be an opt in critical section
#pragma omp critical
        thread_rows.push_back(std::make_pair(thread_id, i*chunkSize));

        Sobel(i);
    }

}
/* **************** Change the function below if you need to ***************** */

int main(int argc, char* argv[]) {

    if (argc != 5) {
        std::cout << "ERROR: Incorrect number of arguments. Format is: <Input image filename> <Output image filename> <Chunk size> <a1/a2>" << std::endl;
        return 0;
    }
 
    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cout << "ERROR: Could not open file " << argv[1] << std::endl;
        return 0;
    }
    chunkSize  = std::atoi(argv[3]);

    // std::cout << "Detect edges in " << argv[1] << " using OpenMP threads" << std::endl;

    /* ******Reading image into 2-D array below******** */

    std::string workString;
    /* Remove comments '#' and check image format */ 
    while (std::getline(file,workString)) {
        if (workString.at(0) != '#') {
            if (workString.at(1) != '2'){
                std::cout << "Input image is not a valid PGM image" << std::endl;
                return 0;
            } else {
                break;
            }       
        } else {
            continue;
        }
    }
    /* Check image size */ 
    while (std::getline(file,workString)) {
        if (workString.at(0) != '#') {
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
    while (std::getline(file,workString)) {
        if (workString.at(0) != '#' ){
            std::stringstream stream(workString);
            stream >> image_maxShades;
            break;
        }
    }

    /* Fill input image matrix */ 
    int pixel_val;
    for (int i = 0; i < image_height; ++i) {
        if (std::getline(file,workString) && workString.at(0) != '#' ) {
            std::stringstream stream(workString);
            for (int j = 0; j < image_width; ++j) {
                if(!stream) break;
                stream >> pixel_val;
                inputImage[i][j] = pixel_val;
            }
        }
    }

    /************ Set up Sobel *********/
    /* 3x3 Sobel mask for X Dimension. */
    maskX[0][0] = -1; maskX[0][1] = 0; maskX[0][2] = 1;
    maskX[1][0] = -2; maskX[1][1] = 0; maskX[1][2] = 2;
    maskX[2][0] = -1; maskX[2][1] = 0; maskX[2][2] = 1;
    /* 3x3 Sobel mask for Y Dimension. */
    maskY[0][0] = 1; maskY[0][1] = 2; maskY[0][2] = 1;
    maskY[1][0] = 0; maskY[1][1] = 0; maskY[1][2] = 0;
    maskY[2][0] = -1; maskY[2][1] = -2; maskY[2][2] = -1;

    /************ Call functions to process image *********/
    std::string opt = argv[4];
    if (!opt.compare("a1")) {    
        double dtime_static = omp_get_wtime();
        compute_sobel_static();
        dtime_static = omp_get_wtime() - dtime_static;
        std::cout << "Static Method Time: " << dtime_static << " seconds\n";
    } else {
        double dtime_dyn = omp_get_wtime();
        compute_sobel_dynamic();
        dtime_dyn = omp_get_wtime() - dtime_dyn;
        std::cout << "Dynamic Method Time: " << dtime_dyn << " seconds\n";
    }

    /* ********Start writing output to your file************ */
    std::ofstream ofile(argv[2]);
    if (ofile.is_open()) {
        ofile << "P2" << "\n" << image_width << " " << image_height << "\n" << image_maxShades << "\n";
        for( int i = 0; i < image_height; ++i) {
            for( int j = 0; j < image_width; ++j) {
                ofile << outputImage[i][j] << " ";
            }
            ofile << "\n";
        }
    } else {
        std::cout << "ERROR: Could not open output file " << argv[2] << std::endl;
        return 0;
    }

    int total_thrd = thread_rows.size();
    for (int thrd_i = 0; thrd_i < total_thrd; ++thrd_i)
        std::cout << "Thread " << thread_rows[thrd_i].first << " -> Processing Chunk starting at Row " << thread_rows[thrd_i].second << std::endl;

    return 0;
}