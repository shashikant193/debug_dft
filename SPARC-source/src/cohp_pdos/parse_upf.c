/**
 * @file    projections_atomic.c
 * @brief   This file contains functions for nonlocal components.
 *
 * @authors Shashikant Kumar <shashikanthome@gmail.com>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
/* BLAS routines */
#ifdef USE_MKL
    #include <mkl.h> // for cblas_* functions
#else
    #include <cblas.h>
#endif

#include "projections_atomic.h"
#include "tools.h"
#include "isddft.h"
#include "initialization.h"
#include "cyclix_tools.h"
#include "parse_upf.h"

#define TEMP_TOL 1e-12

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)>(b)?(b):(a))

#define MAX_LINE_LENGTH 1024
#define MAX_NUMBERS 10000



int extract_num_AO_upf(char *filename) {
    char *key = "number_of_wfc";
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(1);
    }

    char line[1024];
    int extractedNumber = -1; // Default value if the key is not found

    while (fgets(line, sizeof(line), file)) {
        char *pos = strstr(line, key);
        if (pos) {
            // Find the key and extract the number
            sscanf(pos, "%*[^=]=\"%d\"", &extractedNumber);
            break; // Exit the loop after finding the key
        }
    }

    fclose(file);
    return extractedNumber;
}

void read_grid_AO_upf(const char *filename, double *numbers, int *count) {

    char *blockTag = "<PP_R";
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(1);
    }

    char line[1024];
    int insideBlock = 0;  // Flag to track if we're inside the target block
    *count = 0;           // Initialize count of numbers to zero

    while (fgets(line, sizeof(line), file)) {
        // Check for block start
        if (strstr(line, blockTag)) {
            insideBlock = 1;
            continue; // Skip to the next line
        }

        // Check for block end
        if (insideBlock && strstr(line, "</PP_R>")) {
            insideBlock = 0;
            break; // Stop processing after the block ends
        }

        // Extract numbers if inside the block
        if (insideBlock) {
            char *token = strtok(line, " \t\n");
            while (token != NULL) {
                double num;
                if (sscanf(token, "%lf", &num) == 1) {
                    numbers[(*count)++] = num;
                }
                token = strtok(NULL, " \t\n");
            }
        }
    }

    fclose(file);
}

void parseBlocks_upf(const char *filename, int numBlocks, Rnl_obj *blocks) {

    FILE *fid=fopen("debug.txt","w");
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE_LENGTH];
    int currentBlock = -1;
    int readingBlock = 0;
    int readingNumbers = 0;

    while (fgets(line, MAX_LINE_LENGTH, file)) {
        // Check for block start
        if (strstr(line, "<PP_CHI.")) {
            if (currentBlock + 1 >= numBlocks) {
                fprintf(stderr, "Error: More blocks found than expected.\n");
                fclose(file);
                exit(1);
            }
            currentBlock++;
            readingBlock = 1;
            readingNumbers = 0;
            blocks[currentBlock].num_count = 0;
            continue;
        }

        // If inside a block, check for metadata
        if (readingBlock) {
            char *labelPos = strstr(line, "label=");
            if (labelPos) {
                sscanf(labelPos, "label=\"%[^\"]\"", blocks[currentBlock].label);
                continue;
            }

            char *lPos = strstr(line, "l=");
            if (lPos) {
                sscanf(lPos, "l=\"%d\"", &blocks[currentBlock].l);
                readingNumbers = 1;
                continue;
            }
        }

        // If inside a block, check for numeric data
        if (readingBlock && readingNumbers) {
            
            char *token = strtok(line, " \t\n");

            while (token != NULL) {
                double num;
                if (sscanf(token, "%lf", &num) == 1) {
                    blocks[currentBlock].values[blocks[currentBlock].num_count++] = num;
                }
                token = strtok(NULL, " \t\n");
            }
        }

    

        // Check for block end
        if (readingBlock && strstr(line, "</PP_CHI.")) {
            readingBlock = 0;
            readingNumbers = 0;
            continue;
        }

    }

    fclose(file);
    fclose(fid);
}

void read_upf_AO(SPARC_OBJ *pSPARC){

    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    (pSPARC->AO_rad_str).N_rgrid = (int *)malloc(pSPARC->Ntypes*sizeof(int));
    (pSPARC->AO_rad_str).num_orbitals = (int *)malloc(pSPARC->Ntypes*sizeof(int));
    (pSPARC->AO_rad_str).r_grid = (double **)malloc(pSPARC->Ntypes*sizeof(double*));
    (pSPARC->AO_rad_str).Rnl = (Rnl_obj **)malloc(pSPARC->Ntypes*sizeof(Rnl_obj*));

    if (rank==0){
        for (int i = 0; i < pSPARC->Ntypes; i++){
            char *filename = pSPARC->upfname[i];
            int numBlocks = extract_num_AO_upf(filename);
            (pSPARC->AO_rad_str).num_orbitals[i] = numBlocks;
            MPI_Bcast(&(pSPARC->AO_rad_str).num_orbitals[i], 1, MPI_INT, 0, MPI_COMM_WORLD);

            (pSPARC->AO_rad_str).r_grid[i] = (double *)malloc(MAX_NUMBERS*sizeof(double));
            read_grid_AO_upf(filename, (pSPARC->AO_rad_str).r_grid[i], &(pSPARC->AO_rad_str).N_rgrid[i]);

            MPI_Bcast(&(pSPARC->AO_rad_str).N_rgrid[i], 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast((pSPARC->AO_rad_str).r_grid[i], MAX_NUMBERS, MPI_INT, 0, MPI_COMM_WORLD);

            (pSPARC->AO_rad_str).Rnl[i] = (Rnl_obj *)malloc(numBlocks*sizeof(Rnl_obj));
            parseBlocks_upf(filename, numBlocks, (pSPARC->AO_rad_str).Rnl[i]);

           

            for (int j = 0; j < (pSPARC->AO_rad_str).num_orbitals[i]; j++){
                for (int idx_pr = 0; idx_pr < (pSPARC->AO_rad_str).N_rgrid[i]; idx_pr++){
                    // val += (pSPARC->AO_rad_str).Rnl[i][j].values[idx_pr] * (pSPARC->AO_rad_str).Rnl[i][j].values[idx_pr];
                    (pSPARC->AO_rad_str).Rnl[i][j].values[idx_pr] = (pSPARC->AO_rad_str).Rnl[i][j].values[idx_pr]/((pSPARC->AO_rad_str).r_grid[i][idx_pr]);
                }
                (pSPARC->AO_rad_str).Rnl[i][j].values[0] =  (pSPARC->AO_rad_str).Rnl[i][j].values[1];
                // printf("val: %f\n", val*((pSPARC->AO_rad_str).r_grid[i][1]-(pSPARC->AO_rad_str).r_grid[i][0]));
            }

            
            for (int j = 0; j < (pSPARC->AO_rad_str).num_orbitals[i]; j++){
                MPI_Bcast(&(pSPARC->AO_rad_str).Rnl[i][j].label, 20, MPI_CHAR, 0, MPI_COMM_WORLD);
                MPI_Bcast(&(pSPARC->AO_rad_str).Rnl[i][j].l, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&(pSPARC->AO_rad_str).Rnl[i][j].num_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&(pSPARC->AO_rad_str).Rnl[i][j].values, MAX_NUMBERS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        }

    } else {
        for (int i = 0; i < pSPARC->Ntypes; i++){
            (pSPARC->AO_rad_str).r_grid[i] = (double *)malloc(MAX_NUMBERS*sizeof(double));

            MPI_Bcast(&(pSPARC->AO_rad_str).num_orbitals[i], 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&(pSPARC->AO_rad_str).N_rgrid[i], 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast((pSPARC->AO_rad_str).r_grid[i], MAX_NUMBERS, MPI_INT, 0, MPI_COMM_WORLD);

            (pSPARC->AO_rad_str).Rnl[i] = (Rnl_obj *)malloc((pSPARC->AO_rad_str).num_orbitals[i]*sizeof(Rnl_obj));



            for (int j = 0; j < (pSPARC->AO_rad_str).num_orbitals[i]; j++){
                MPI_Bcast(&(pSPARC->AO_rad_str).Rnl[i][j].label, 20, MPI_CHAR, 0, MPI_COMM_WORLD);
                MPI_Bcast(&(pSPARC->AO_rad_str).Rnl[i][j].l, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&(pSPARC->AO_rad_str).Rnl[i][j].num_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&(pSPARC->AO_rad_str).Rnl[i][j].values, MAX_NUMBERS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        }
    }
    
}





// int main() {

//     double *r_grid = (double *)malloc(sizeof(double)*MAX_NUMBERS);
//     int N_rgrid;
//     int numBlocks;
//     char *filename = "Mg.upf";

//     numBlocks = extract_num_AO_upf(filename);
//     Rnl_obj Rnl[numBlocks];

    
//     read_grid_AO_upf(filename, r_grid, &N_rgrid);
//     parseBlocks_upf(filename, numBlocks, Rnl);


//     // Print extracted data
//     for (int i = 0; i < numBlocks; i++) {
//         printf("Block %d:\n", i + 1);
//         printf("  Label: %s\n", Rnl[i].label);
//         printf("  l: %d\n", Rnl[i].l);
//     }

//     return 0;
// }
