#ifndef PARSE_UPF_H
#define PARSE_UPF_H 

#include "isddft.h"



int extract_num_AO_upf(char *filename);

void read_grid_AO_upf(const char *filename, double *numbers, int *count);

void parseBlocks_upf(const char *filename, int numBlocks, Rnl_obj *blocks);

void read_upf_AO(SPARC_OBJ *pSPARC);




#endif // PARSE_UPF_H 