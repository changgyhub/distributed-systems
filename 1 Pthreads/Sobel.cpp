/*
 * Sobel filter on pgm image with pthreads
 * Author: CHANG GAO
 * Development platform: g++ (Ubuntu 6.2.0-3ubuntu11~14.04) 6.2.0
 * Last modified date: 29 Jan 2017
 * Compilation: g++ -Wall -std=c++11 -pthread Sobel.cpp -o Sobel
                ./Sobel <Input image filename> <Output image filename> <Threads#> <Chunk size>
 */

#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <mutex>
#include <thread>

/* Global variables, Look at their usage in main() */
int image_height;
int image_width;
int image_maxShades;
int inputImage[1000][1000];
int outputImage[1000][1000];
int num_threads; 
int chunkSize;
int maxChunk;
int chunkcnt;
int maskX[3][3];
int maskY[3][3];
std::mutex chunk_mutex;

/* **************** functions ***************** */
void calcmask(int thread_num){
    int sumx, sumy, sum;
    do{
        // lock the mutex and test if there are remaining chunks
        {
            std::lock_guard<std::mutex> lock(chunk_mutex);
            if (chunkcnt >= maxChunk-1) return;
            ++chunkcnt;
            fprintf(stdout, "Thread %d process chunk %d\n", thread_num, chunkcnt);
        }
        // start masking
        for(int x = chunkSize*chunkcnt; x < chunkSize*(chunkcnt+1); ++x){
            for(int y = 0; y < image_width; ++y){
                sumx = 0;
                sumy = 0;
                /* For handling image boundaries */
                if( x <= 0 || x >= (image_height-1) || y <= 0 || y >= (image_width-1)) sum = 0;
                else{
                    /* Gradient calculation*/
                    for(int i = -1; i <= 1; ++i) {
                        for(int j = -1; j <= 1; ++j){
                            sumx += inputImage[x+i][y+j] * maskX[i+1][j+1];
                            sumy += inputImage[x+i][y+j] * maskY[i+1][j+1];
                        }
                    }
                    /* Gradient magnitude */
                    sum = (abs(sumx) + abs(sumy));
                }
                outputImage[x][y] = sum < 0 ? 0 : sum > 255 ? 255 : sum;
            }
        }
    } while(1);
}


void dispatch_threads(){
    /* 3x3 Sobel mask for X Dimension. */
    maskX[0][0] = -1; maskX[0][1] = 0; maskX[0][2] = 1;
    maskX[1][0] = -2; maskX[1][1] = 0; maskX[1][2] = 2;
    maskX[2][0] = -1; maskX[2][1] = 0; maskX[2][2] = 1;
    /* 3x3 Sobel mask for Y Dimension. */
    maskY[0][0] = 1; maskY[0][1] = 2; maskY[0][2] = 1;
    maskY[1][0] = 0; maskY[1][1] = 0; maskY[1][2] = 0;
    maskY[2][0] = -1; maskY[2][1] = -2; maskY[2][2] = -1;
    // setup chunks
    chunkcnt = -1;
    // create threads
    std::vector<std::thread> threads;
    for(int i = 0; i < num_threads; ++i) threads.emplace_back(calcmask, i);
    for(int i = 0; i < num_threads; ++i) threads[i].join();
}

/* **************** main ***************** */

int main(int argc, char** argv){
    if(argc != 5){
        std::cout << "ERROR: Incorrect number of arguments. Format is: <Input image filename> <Output image filename> <Threads#> <Chunk size>" << std::endl;
        return 0;
    }
 
    std::ifstream file(argv[1]);
    if(!file.is_open()){
        std::cout << "ERROR: Could not open file " << argv[1] << std::endl;
        return 0;
    }
    num_threads = std::atoi(argv[3]);
    chunkSize  = std::atoi(argv[4]);

    std::cout << "Detect edges in " << argv[1] << " using " << num_threads << " threads" << std::endl;

    /* ******Reading image into 2-D array below******** */

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
        } else {
            continue;
        }
    }
    /* Check image size */ 
    while(std::getline(file,workString)){
        if( workString.at(0) != '#' ){
            std::stringstream stream(workString);
            int n;
            stream >> n;
            image_width = n;
            stream >> n;
            image_height = n;
            break;
        } else {
            continue;
        }
    }

    /* maxChunk is total number of chunks to process */
    maxChunk = ceil((float)image_height/chunkSize);

    /* Check image max shades */ 
    while(std::getline(file,workString)){
        if( workString.at(0) != '#' ){
            std::stringstream stream(workString);
            stream >> image_maxShades;
            break;
        } else {
            continue;
        }
    }
    /* Fill input image matrix */ 
    int pixel_val;
    for( int i = 0; i < image_height; i++ ){
        if( std::getline(file,workString) && workString.at(0) != '#' ){
            std::stringstream stream(workString);
            for( int j = 0; j < image_width; j++ ){
                if( !stream )
                    break;
                stream >> pixel_val;
                inputImage[i][j] = pixel_val;
            }
        } else {
            continue;
        }
    }

    /************ Function that creates threads and manage dynamic allocation of chunks *********/
    dispatch_threads();

    /* ********Start writing output to your file************ */
    std::ofstream ofile(argv[2]);
    if( ofile.is_open() ){
        ofile << "P2" << "\n" << image_width << " " << image_height << "\n" << image_maxShades << "\n";
        for( int i = 0; i < image_height; i++ )
        {
            for( int j = 0; j < image_width; j++ ){
                ofile << outputImage[i][j] << " ";
            }
            ofile << "\n";
        }
    } else {
        std::cout << "ERROR: Could not open output file " << argv[2] << std::endl;
        return 0;
    }
    return 0;
}