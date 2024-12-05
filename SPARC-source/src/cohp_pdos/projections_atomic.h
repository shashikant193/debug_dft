/**
 * @file    projections_atomic.h
 * @brief   This file contains the function declarations for eigen-solvers.
 *
 * @authors SHashikant Kumar <shashikanthome@gmail.com>

**/
#ifndef PROJECTIONS_ATOMIC_H
#define PROJECTIONS_ATOMIC_H 

#include "isddft.h"
#include <stdbool.h>


void GetInfluencingAtoms_AtomicOrbitals(SPARC_OBJ *pSPARC, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, int *DMVertices, MPI_Comm comm);

void CalculateAtomicOrbitals(SPARC_OBJ *pSPARC, AO_OBJ *AO_str, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, int *DMVertices, MPI_Comm comm);

void Calculate_SplineDerivAORadFun(SPARC_OBJ *pSPARC);

void CalculateAtomicOrbitals_kpt(SPARC_OBJ *pSPARC, AO_OBJ *AO_str, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, int *DMVertices, MPI_Comm comm);

void Calculate_Overlap_AO(SPARC_OBJ *pSPARC, AO_OBJ *AO_str, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, double ******OIJ_local_images, int *DMVertices, MPI_Comm comm);

void Calculate_Overlap_AO_kpt(SPARC_OBJ *pSPARC, AO_OBJ *AO_str, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, int kpt, double _Complex ******OIJ_local_images, int *DMVertices, MPI_Comm comm);

void get_common_idx(int *x, int *y, int nx, int ny, int *n_common, int *nx_common, int *ny_common);


void CalculateAOInnerProductIndex(SPARC_OBJ *pSPARC);

void AO_psi_mult(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, 
                  AO_OBJ *AO_str, int ncol, double *x, int ldi, double *Hx, int i_spin, int ldo,  MPI_Comm comm);

void AO_psi_mult_kpt(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, 
                      AO_OBJ *AO_str, int ncol, double _Complex *x, int ldi, double _Complex *Hx, int i_spin, int ldo, int kpt, MPI_Comm comm);

void get_DOS(SPARC_OBJ *pSPARC, double **DOS, int NDOS, int N_AO);

void get_PDOS(SPARC_OBJ *pSPARC, double **DOS, int NDOS, int N_AO);

void get_cohp(SPARC_OBJ *pSPARC, double **DOS, int NDOS, int N_AO);

void print_DOS_COHP(SPARC_OBJ *pSPARC);

#endif // PARSE_UPF_H 