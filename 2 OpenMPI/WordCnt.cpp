/*
 * Word count in a file with OpenMPI
 * Author: CHANG GAO
 * Development platform: g++ (Ubuntu 5.4.1-2ubuntu1~14.04) 5.4.1 20160904
 * Last modified date: 14 Feb 2017
 * Compilation: mpic++ -std=c++11 WordCnt.cpp -o WordCnt
                mpirun -np <num_of_process> ./WordCnt <filename> <word> <b1/b2>
 */
#include "mpi.h"
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cctype>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
const static int ARRAY_SIZE = 130000;
using Lines = char[ARRAY_SIZE][16];
using Words = char[16];

// To remove punctuations
struct letter_only: std::ctype<char> {
    letter_only(): std::ctype<char>(get_table()) {}

    static std::ctype_base::mask const* get_table(){
        static std::vector<std::ctype_base::mask> 
            rc(std::ctype<char>::table_size,std::ctype_base::space);

        std::fill(&rc['A'], &rc['z'+1], std::ctype_base::alpha);
        return &rc[0];
    }
};

void DoOutput(std::string word, int result) {
    std::cout << "Word Frequency: " << word << " -> " << result << std::endl;
}

int search_cnt(Words* words, int num, const char* target){
    int local_cnt = 0;
    for (int i = 0; i < num && words[i]; ++i){
        if (!strcmp(words[i], target)) ++local_cnt;
    }
    return local_cnt;
}

//***************** Add your functions here *********************

int main(int argc, char* argv[]) {
    int processId, num_processes, total_cnt = 0;
    int *to_return = NULL;
    double start_time, end_time;
 
    // Setup MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
 
    // Three arguments: <input file> <search word> <part B1 or part B2 to execute>
    if (argc != 4) {
        if(processId == 0) {
            std::cout << "ERROR: Incorrect number of arguments. Format is: <filename> <word> <b1/b2>" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }
    const char* word = argv[2];
 
    Lines lines;
    int line_cnt = 0;
    // Read the input file and put words into char array(lines)
    if (processId == 0) {
        std::ifstream file;
        file.imbue(std::locale(std::locale(), new letter_only()));
        file.open(argv[1]);
        std::string workString;
        while(file >> workString){
            memset(lines[line_cnt], '\0', 16);
            memcpy(lines[line_cnt++], workString.c_str(), workString.length());
        }
    }
    
//  ***************** Add code as per your requirement below ***************** 
    
    start_time = MPI_Wtime();

    if(!strcmp(argv[3], "b1") || !strcmp(argv[3], "b2")) {
        int* words_to_scatter = NULL;
        int words_to_read;
        
        if (processId == 0){
            words_to_scatter = new int[num_processes];
            int words_per_process = ceil(line_cnt*1.0/num_processes);
            for (int i = 0; i < num_processes; ++i){
                words_to_scatter[i] = words_per_process;
            }           
        }
        // scatter lines info 
        MPI_Scatter(
            words_to_scatter, // void* send_data
            1, // int send_count
            MPI_INT, // MPI_Datatype send_datatype
            &words_to_read, // void* recv_data
            1, // int recv_count
            MPI_INT, // MPI_Datatype recv_datatype
            0, // int root
            MPI_COMM_WORLD // MPI_Comm communicator
        ); 

        Words* wordsChunk = new Words[words_to_read];

        // scatter lines 
        MPI_Scatter(
            &lines[0][0], // void* send_data
            words_to_read*16, // int send_count
            MPI_CHAR, // MPI_Datatype send_datatype
            &wordsChunk[0][0], // void* recv_data
            words_to_read*16, // int recv_count
            MPI_CHAR, // MPI_Datatype recv_datatype
            0, // int root
            MPI_COMM_WORLD // MPI_Comm communicator
        );

        // start searching
        int wordChunkCnt = search_cnt(wordsChunk, words_to_read, word);

        if (!strcmp(argv[3], "b1")){
            // Using Reduction
            MPI_Reduce(
                &wordChunkCnt, // void* send_data
                &total_cnt, // void* recv_data
                1, // int count
                MPI_INT, // MPI_Datatype datatype
                MPI_SUM, // MPI_Op op
                0, // int root
                MPI_COMM_WORLD
            ); // MPI_Comm communicator
        } else {
            // Using Ring topology
            if (num_processes == 1){
                total_cnt = wordChunkCnt;
            } else if (processId == 0){
                MPI_Send(&wordChunkCnt, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&total_cnt, 1, MPI_INT, num_processes-1,
                    MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(&total_cnt, 1, MPI_INT, processId-1,
                    MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                total_cnt += wordChunkCnt;
                MPI_Send(&total_cnt, 1, MPI_INT, (processId+1)%(num_processes), 0, MPI_COMM_WORLD);
            }
        }
        
        // output result
        if (processId == 0) {
            DoOutput(std::string(word), total_cnt);
            end_time = MPI_Wtime();
            std::cout << "Time: " << ((double)end_time-start_time) << std::endl;
            delete [] words_to_scatter;
        }
        delete [] wordsChunk;
    }

    MPI_Finalize();
    return 0;
}