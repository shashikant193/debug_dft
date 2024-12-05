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
#include <time.h>
#include <stdbool.h>
/* BLAS routines */
#ifdef USE_MKL
    #include <mkl.h> // for cblas_* functions
#else
    #include <cblas.h>
#endif

#include "projections_atomic.h"
#include "parse_upf.h"
#include "tools.h"
#include "isddft.h"
#include "initialization.h"
#include "cyclix_tools.h"

#define TEMP_TOL 1e-12

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)>(b)?(b):(a))
#define HASH_SIZE 10007  // A prime number for hash table size


/**
 * @brief   Find the list of all atoms that influence the processor 
 *          domain in psi-domain.
 */
void GetInfluencingAtoms_AtomicOrbitals(SPARC_OBJ *pSPARC, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, int *DMVertices, MPI_Comm comm) 
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) printf("Finding atoms that has nonlocal influence on the local process domain ... \n");
#endif
    // processors that are not in the dmcomm will remain idle
    // on input make comm of processes with bandcomm_index or kptcomm_index MPI_COMM_NULL!
    if (comm == MPI_COMM_NULL) {
        return; 
    }

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    int nproc_comm, rank_comm;
    MPI_Comm_size(comm, &nproc_comm);
    MPI_Comm_rank(comm, &rank_comm);
    
    double DMxs, DMxe, DMys, DMye, DMzs, DMze;
    double Lx, Ly, Lz, rc, x0, y0, z0, x0_i, y0_i, z0_i, x2, y2, z2, r2, rc2, rcbox_x, rcbox_y, rcbox_z, x, y, z;
    int count_overlap_nloc, count_overlap_nloc_sphere, ityp, i, j, k, count, i_DM, j_DM, k_DM, 
        iat, atmcount, atmcount2, DMnx, DMny;
    int pp, qq, rr, ppmin, ppmax, qqmin, qqmax, rrmin, rrmax;
    int rc_xl, rc_xr, rc_yl, rc_yr, rc_zl, rc_zr, ndc;

    

    // Lattice vectors
    double a1[3] = {pSPARC->LatUVec[0], pSPARC->LatUVec[1], pSPARC->LatUVec[2]};
    double a2[3] = {pSPARC->LatUVec[3], pSPARC->LatUVec[4], pSPARC->LatUVec[5]};
    double a3[3] = {pSPARC->LatUVec[6], pSPARC->LatUVec[7], pSPARC->LatUVec[8]};

    // Reciprocal lattice vectors (not-normalized) It is just b1 = a2 \cross a3 and so on
    double b1[3] = {a2[1]*a3[2] - a2[2]*a3[1], -1.0*(a2[0]*a3[2]-a2[2]*a3[0]), a2[0]*a3[1]-a2[1]*a3[0]};
    double b2[3] = {a3[1]*a1[2] - a3[2]*a1[1], -1.0*(a3[0]*a1[2]-a3[2]*a1[0]), a3[0]*a1[1]-a3[1]*a1[0]};
    double b3[3] = {a1[1]*a2[2] - a1[2]*a2[1], -1.0*(a1[0]*a2[2]-a1[2]*a2[0]), a1[0]*a2[1]-a1[1]*a2[0]};

    // vector norm of a1,a2,a3, and b1,b2,b3
    double norm_b1 = sqrt(b1[0]*b1[0] + b1[1]*b1[1] + b1[2]*b1[2]);
    double norm_b2 = sqrt(b2[0]*b2[0] + b2[1]*b2[1] + b2[2]*b2[2]);
    double norm_b3 = sqrt(b3[0]*b3[0] + b3[1]*b3[1] + b3[2]*b3[2]);
    
    double norm_a1 = sqrt(a1[0]*a1[0] + a1[1]*a1[1] + a1[2]*a1[2]);
    double norm_a2 = sqrt(a2[0]*a2[0] + a2[1]*a2[1] + a2[2]*a2[2]);
    double norm_a3 = sqrt(a3[0]*a3[0] + a3[1]*a3[1] + a3[2]*a3[2]);

    // unit vectors of b
    double b1_uvec[3] = {b1[0]/norm_b1, b1[1]/norm_b1, b1[2]/norm_b1};
    double b2_uvec[3] = {b2[0]/norm_b2, b2[1]/norm_b2, b2[2]/norm_b2};
    double b3_uvec[3] = {b3[0]/norm_b3, b3[1]/norm_b3, b3[2]/norm_b3};

    // dot product of a with b unit vecs
    double a1_tilde = a1[0]*b1_uvec[0] + a1[1]*b1_uvec[1] + a1[2]*b1_uvec[2];
    double a2_tilde = a2[0]*b2_uvec[0] + a2[1]*b2_uvec[1] + a2[2]*b2_uvec[2];
    double a3_tilde = a3[0]*b3_uvec[0] + a3[1]*b3_uvec[1] + a3[2]*b3_uvec[2];




    Lx = pSPARC->range_x;
    Ly = pSPARC->range_y;
    Lz = pSPARC->range_z;
    
    // TODO: notice the difference here from rb-domain, since nonlocal projectors decay drastically, using the fd-nodes as edges are more appropriate
    DMxs = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;
    DMxe = pSPARC->xin + (DMVertices[1]) * pSPARC->delta_x; // note that this is not the actual edge, add BCx to get actual domain edge
    DMys = DMVertices[2] * pSPARC->delta_y;
    DMye = (DMVertices[3]) * pSPARC->delta_y; // note that this is not the actual edge, add BCx to get actual domain edge
    DMzs = DMVertices[4] * pSPARC->delta_z;
    DMze = (DMVertices[5]) * pSPARC->delta_z; // note that this is not the actual edge, add BCx to get actual domain edge
    
    // number of nodes in the local distributed domain
    DMnx = DMVertices[1] - DMVertices[0] + 1;
    DMny = DMVertices[3] - DMVertices[2] + 1;



    // TODO: in the future, it's better to save only the atom types that have influence on the local domain
    // *Atom_Influence_AO = (ATOM_NLOC_INFLUENCE_OBJ *)malloc(sizeof(ATOM_NLOC_INFLUENCE_OBJ) * pSPARC->Ntypes);

    
    ATOM_NLOC_INFLUENCE_OBJ Atom_Influence_temp;

    // find which atoms have nonlocal influence on the distributed domain owned by current process
    atmcount = 0; atmcount2 = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        // rc = 0.0;
        // // find max rc 
        // for (i = 0; i <= pSPARC->psd[ityp].lmax; i++) {
        //     rc = max(rc, pSPARC->psd[ityp].rc[i]);
        // }
        rc = min(10.0, (pSPARC->AO_rad_str).r_grid[ityp][ (pSPARC->AO_rad_str).N_rgrid[ityp]-1]);  // 10 Bohr is the cutoff for the atomic orbitals


        rc2 = rc * rc;
        
        if(pSPARC->cell_typ == 0) {
            rcbox_x = rcbox_y = rcbox_z = rc;            
        } else {
            rcbox_x = ceil(rc/a1_tilde) * norm_a1;
            rcbox_y = ceil(rc/a2_tilde) * norm_a2;
            rcbox_z = ceil(rc/a3_tilde) * norm_a3;
        }

        // first loop over all atoms of each type to find number of influencing atoms
        count_overlap_nloc = 0;
        for (i = 0; i < pSPARC->nAtomv[ityp]; i++) {
            // get atom positions
            x0 = pSPARC->atom_pos[3*atmcount];
            y0 = pSPARC->atom_pos[3*atmcount+1];
            z0 = pSPARC->atom_pos[3*atmcount+2];
            atmcount++;
            ppmin = ppmax = qqmin = qqmax = rrmin = rrmax = 0;
            if (pSPARC->BCx == 0) {
                if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {
                    // rcut_x is the real ruct in x direction. 
                    double rcut_x = pSPARC->pSQ->nloc[0] * pSPARC->delta_x;
                    ppmax = floor((rcbox_x + Lx - x0 + rcut_x) / Lx + TEMP_TOL);
                    ppmin = -floor((rcbox_x + x0 + rcut_x) / Lx + TEMP_TOL);    
                } else {
                    ppmax = floor((rcbox_x + Lx - x0) / Lx + TEMP_TOL);
                    ppmin = -floor((rcbox_x + x0) / Lx + TEMP_TOL);
                }
            }
            if (pSPARC->BCy == 0) {
                if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {
                    double rcut_y = pSPARC->pSQ->nloc[1] * pSPARC->delta_y;
                    qqmax = floor((rcbox_y + Ly - y0 + rcut_y) / Ly + TEMP_TOL);
                    qqmin = -floor((rcbox_y + y0 + rcut_y) / Ly + TEMP_TOL);
                } else {
                    qqmax = floor((rcbox_y + Ly - y0) / Ly + TEMP_TOL);
                    qqmin = -floor((rcbox_y + y0) / Ly + TEMP_TOL);
                }
            }
            if (pSPARC->BCz == 0) {
                if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {
                    double rcut_z = pSPARC->pSQ->nloc[2] * pSPARC->delta_z;
                    rrmax = floor((rcbox_z + Lz - z0 + rcut_z) / Lz + TEMP_TOL);
                    rrmin = -floor((rcbox_z + z0 + rcut_z) / Lz + TEMP_TOL);
                } else {
                    rrmax = floor((rcbox_z + Lz - z0) / Lz + TEMP_TOL);
                    rrmin = -floor((rcbox_z + z0) / Lz + TEMP_TOL);
                }
            }

            // check how many of it's images interacts with the local distributed domain
            for (rr = rrmin; rr <= rrmax; rr++) {
                z0_i = z0 + Lz * rr; // z coord of image atom
                if ((z0_i < DMzs - rcbox_z) || (z0_i >= DMze + rcbox_z)) continue;
                for (qq = qqmin; qq <= qqmax; qq++) {
                    y0_i = y0 + Ly * qq; // y coord of image atom
                    if ((y0_i < DMys - rcbox_y) || (y0_i >= DMye + rcbox_y)) continue;
                    for (pp = ppmin; pp <= ppmax; pp++) {
                        x0_i = x0 + Lx * pp; // x coord of image atom
                        if ((x0_i < DMxs - rcbox_x) || (x0_i >= DMxe + rcbox_x)) continue;
                        count_overlap_nloc++;
                    }
                }
            }
        } // end for loop over atoms of each type, for the first time
        
        Atom_Influence_temp.n_atom = count_overlap_nloc;
        Atom_Influence_temp.coords = (double *)malloc(sizeof(double) * count_overlap_nloc * 3);
        Atom_Influence_temp.atom_index = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_temp.xs = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_temp.ys = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_temp.zs = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_temp.xe = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_temp.ye = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_temp.ze = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_temp.ndc = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_temp.grid_pos = (int **)malloc(sizeof(int*) * count_overlap_nloc);

        // when there's no atom of this type that have influence, go to next type of atom
        if (Atom_Influence_temp.n_atom == 0) {
            Atom_Influence_AO[ityp].n_atom = 0;
            atmcount2 = atmcount;
            free(Atom_Influence_temp.coords);
            free(Atom_Influence_temp.atom_index);
            free(Atom_Influence_temp.xs);
            free(Atom_Influence_temp.ys);
            free(Atom_Influence_temp.zs);
            free(Atom_Influence_temp.xe);
            free(Atom_Influence_temp.ye);
            free(Atom_Influence_temp.ze);
            free(Atom_Influence_temp.ndc);
            free(Atom_Influence_temp.grid_pos);
            continue;
        }
        
        // loop over atoms of this type again to find overlapping region and atom info
        count_overlap_nloc = 0;
        count_overlap_nloc_sphere = 0;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            // get atom positions
            x0 = pSPARC->atom_pos[3*atmcount2];
            y0 = pSPARC->atom_pos[3*atmcount2+1];
            z0 = pSPARC->atom_pos[3*atmcount2+2];
            atmcount2++;
            ppmin = ppmax = qqmin = qqmax = rrmin = rrmax = 0;
            if (pSPARC->BCx == 0) {
                if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {
                    double rcut_x = pSPARC->pSQ->nloc[0] * pSPARC->delta_x;
                    ppmax = floor((rcbox_x + Lx - x0 + rcut_x) / Lx + TEMP_TOL);
                    ppmin = -floor((rcbox_x + x0 + rcut_x) / Lx + TEMP_TOL);    
                } else {
                    ppmax = floor((rcbox_x + Lx - x0) / Lx + TEMP_TOL);
                    ppmin = -floor((rcbox_x + x0) / Lx + TEMP_TOL);
                }
            }
            if (pSPARC->BCy == 0) {
                if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {
                    double rcut_y = pSPARC->pSQ->nloc[1] * pSPARC->delta_y;
                    qqmax = floor((rcbox_y + Ly - y0 + rcut_y) / Ly + TEMP_TOL);
                    qqmin = -floor((rcbox_y + y0 + rcut_y) / Ly + TEMP_TOL);
                } else {
                    qqmax = floor((rcbox_y + Ly - y0) / Ly + TEMP_TOL);
                    qqmin = -floor((rcbox_y + y0) / Ly + TEMP_TOL);
                }
            }
            if (pSPARC->BCz == 0) {
                if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {
                    double rcut_z = pSPARC->pSQ->nloc[2] * pSPARC->delta_z;
                    rrmax = floor((rcbox_z + Lz - z0 + rcut_z) / Lz + TEMP_TOL);
                    rrmin = -floor((rcbox_z + z0 + rcut_z) / Lz + TEMP_TOL);
                } else {
                    rrmax = floor((rcbox_z + Lz - z0) / Lz + TEMP_TOL);
                    rrmin = -floor((rcbox_z + z0) / Lz + TEMP_TOL);
                }
            }
            
            // check if this image interacts with the local distributed domain
            for (rr = rrmin; rr <= rrmax; rr++) {
                z0_i = z0 + Lz * rr; // z coord of image atom
                if ((z0_i < DMzs - rcbox_z) || (z0_i >= DMze + rcbox_z)) continue;
                for (qq = qqmin; qq <= qqmax; qq++) {
                    y0_i = y0 + Ly * qq; // y coord of image atom
                    if ((y0_i < DMys - rcbox_y) || (y0_i >= DMye + rcbox_y)) continue;
                    for (pp = ppmin; pp <= ppmax; pp++) {
                        x0_i = x0 + Lx * pp; // x coord of image atom
                        if ((x0_i < DMxs - rcbox_x) || (x0_i >= DMxe + rcbox_x)) continue;
                        
                        // store coordinates of the overlapping atom
                        Atom_Influence_temp.coords[count_overlap_nloc*3  ] = x0_i;
                        Atom_Influence_temp.coords[count_overlap_nloc*3+1] = y0_i;
                        Atom_Influence_temp.coords[count_overlap_nloc*3+2] = z0_i;
                        
                        // record the original atom index this image atom corresponds to
                        Atom_Influence_temp.atom_index[count_overlap_nloc] = atmcount2-1;
                        
                        // find start & end nodes of the rc-region of the image atom
                        // This way, we try to make sure all points inside rc-region
                        // is strictly less that rc distance away from the image atom
                        rc_xl = ceil( (x0_i - pSPARC->xin - rcbox_x)/pSPARC->delta_x);
                        rc_xr = floor((x0_i - pSPARC->xin + rcbox_x)/pSPARC->delta_x);
                        rc_yl = ceil( (y0_i - rcbox_y)/pSPARC->delta_y);
                        rc_yr = floor((y0_i + rcbox_y)/pSPARC->delta_y);
                        rc_zl = ceil( (z0_i - rcbox_z)/pSPARC->delta_z);
                        rc_zr = floor((z0_i + rcbox_z)/pSPARC->delta_z);
                        
                        // TODO: check if rc-region is out of fundamental domain for BC == 1!
                        // find overlap of rc-region of the image and the local dist. domain
                        Atom_Influence_temp.xs[count_overlap_nloc] = max(DMVertices[0], rc_xl);
                        Atom_Influence_temp.xe[count_overlap_nloc] = min(DMVertices[1], rc_xr);
                        Atom_Influence_temp.ys[count_overlap_nloc] = max(DMVertices[2], rc_yl);
                        Atom_Influence_temp.ye[count_overlap_nloc] = min(DMVertices[3], rc_yr);
                        Atom_Influence_temp.zs[count_overlap_nloc] = max(DMVertices[4], rc_zl);
                        Atom_Influence_temp.ze[count_overlap_nloc] = min(DMVertices[5], rc_zr);

                        // find the spherical rc-region
                        ndc = (Atom_Influence_temp.xe[count_overlap_nloc] - Atom_Influence_temp.xs[count_overlap_nloc] + 1)
                            * (Atom_Influence_temp.ye[count_overlap_nloc] - Atom_Influence_temp.ys[count_overlap_nloc] + 1)
                            * (Atom_Influence_temp.ze[count_overlap_nloc] - Atom_Influence_temp.zs[count_overlap_nloc] + 1);
                        
                        // first allocate memory for the rectangular rc-region, resize later to the spherical rc-region
                        Atom_Influence_temp.grid_pos[count_overlap_nloc] = (int *)malloc(sizeof(int) * ndc);
                        count = 0;
                        for (k = Atom_Influence_temp.zs[count_overlap_nloc]; k <= Atom_Influence_temp.ze[count_overlap_nloc]; k++) {
                            k_DM = k - DMVertices[4];
                            z = k * pSPARC->delta_z;
                            for (j = Atom_Influence_temp.ys[count_overlap_nloc]; j <= Atom_Influence_temp.ye[count_overlap_nloc]; j++) {
                                j_DM = j - DMVertices[2];
                                y = j * pSPARC->delta_y;
                                for (i = Atom_Influence_temp.xs[count_overlap_nloc]; i <= Atom_Influence_temp.xe[count_overlap_nloc]; i++) {
                                    i_DM = i - DMVertices[0];
                                    x = pSPARC->xin + i * pSPARC->delta_x;
                                    CalculateDistance(pSPARC, x, y, z, x0_i, y0_i, z0_i, &r2);
                                    r2 *= r2;

                                    // What is the need for this if condition below?
                                    if (r2 <= rc2) {
                                        Atom_Influence_temp.grid_pos[count_overlap_nloc][count] = k_DM * (DMnx * DMny) + j_DM * DMnx + i_DM;
                                        count++;
                                    }
                                }
                            }
                        }
                        // TODO: in some cases count is 0! check if ndc is 0 and remove those!
                        Atom_Influence_temp.ndc[count_overlap_nloc] = count;
                        count_overlap_nloc++;
                        
                        if (count > 0) {
                            count_overlap_nloc_sphere++;
                        }
                    }
                }
            }
        }
        
        if (count_overlap_nloc_sphere == 0) {
            Atom_Influence_AO[ityp].n_atom = 0;
            atmcount2 = atmcount;
            free(Atom_Influence_temp.coords);
            free(Atom_Influence_temp.atom_index);
            free(Atom_Influence_temp.xs);
            free(Atom_Influence_temp.ys);
            free(Atom_Influence_temp.zs);
            free(Atom_Influence_temp.xe);
            free(Atom_Influence_temp.ye);
            free(Atom_Influence_temp.ze);
            free(Atom_Influence_temp.ndc);
            for (i = 0; i < count_overlap_nloc; i++) {
                free(Atom_Influence_temp.grid_pos[i]);
            }
            free(Atom_Influence_temp.grid_pos);
            continue;
        }

        Atom_Influence_AO[ityp].n_atom = count_overlap_nloc_sphere;
        Atom_Influence_AO[ityp].coords = (double *)malloc(sizeof(double) * count_overlap_nloc_sphere * 3);
        Atom_Influence_AO[ityp].atom_index = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        Atom_Influence_AO[ityp].xs = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        Atom_Influence_AO[ityp].ys = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        Atom_Influence_AO[ityp].zs = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        Atom_Influence_AO[ityp].xe = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        Atom_Influence_AO[ityp].ye = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        Atom_Influence_AO[ityp].ze = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        Atom_Influence_AO[ityp].ndc = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        Atom_Influence_AO[ityp].grid_pos = (int **)malloc(sizeof(int*) * count_overlap_nloc_sphere);
        
        count = 0;
        for (i = 0; i < count_overlap_nloc; i++) {
            if ( Atom_Influence_temp.ndc[i] > 0 ) {
                ndc = Atom_Influence_temp.ndc[i];
                Atom_Influence_AO[ityp].coords[count*3] = Atom_Influence_temp.coords[i*3];
                Atom_Influence_AO[ityp].coords[count*3+1] = Atom_Influence_temp.coords[i*3+1];
                Atom_Influence_AO[ityp].coords[count*3+2] = Atom_Influence_temp.coords[i*3+2];
                Atom_Influence_AO[ityp].atom_index[count] = Atom_Influence_temp.atom_index[i];
                Atom_Influence_AO[ityp].xs[count] = Atom_Influence_temp.xs[i];
                Atom_Influence_AO[ityp].ys[count] = Atom_Influence_temp.ys[i];
                Atom_Influence_AO[ityp].zs[count] = Atom_Influence_temp.zs[i];
                Atom_Influence_AO[ityp].xe[count] = Atom_Influence_temp.xe[i];
                Atom_Influence_AO[ityp].ye[count] = Atom_Influence_temp.ye[i];
                Atom_Influence_AO[ityp].ze[count] = Atom_Influence_temp.ze[i];
                Atom_Influence_AO[ityp].ndc[count] = Atom_Influence_temp.ndc[i];

                Atom_Influence_AO[ityp].grid_pos[count] = (int *)malloc(sizeof(int) * ndc);
                for (j = 0; j < ndc; j++) {
                    Atom_Influence_AO[ityp].grid_pos[count][j] = Atom_Influence_temp.grid_pos[i][j];
                }
                count++;
            }
            free(Atom_Influence_temp.grid_pos[i]);
        }
        
        free(Atom_Influence_temp.coords);
        free(Atom_Influence_temp.atom_index);
        free(Atom_Influence_temp.xs);
        free(Atom_Influence_temp.ys);
        free(Atom_Influence_temp.zs);
        free(Atom_Influence_temp.xe);
        free(Atom_Influence_temp.ye);
        free(Atom_Influence_temp.ze);
        free(Atom_Influence_temp.ndc);
        free(Atom_Influence_temp.grid_pos);
    }
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf(GRN"rank = %d, time for nonlocal influencing atoms: %.3f ms\n"RESET, rank, (t2-t1)*1e3);
#endif
}




/**
 * @brief   Calculate Atomic orbitals
 */
void CalculateAtomicOrbitals(SPARC_OBJ *pSPARC, AO_OBJ *AO_str, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, int *DMVertices, MPI_Comm comm)
{
    // processors that are not in the dmcomm will continue
    if (comm == MPI_COMM_NULL) {
        return; // upon input, make sure process with bandcomm_index < 0 provides MPI_COMM_NULL
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2, t_tot, t_old;

    t_tot = t_old = 0.0;
#ifdef DEBUG
    if (rank == 0) printf("Calculate nonlocal projectors ... \n");
#endif    
    int l, n_ao, np, lcount, lcount2, m, psd_len, col_count, indx, ityp, iat, ipos, ndc, lloc, lmax, DMnx, DMny;
    int i_DM, j_DM, k_DM;
    double x0_i, y0_i, z0_i, *rc_pos_x, *rc_pos_y, *rc_pos_z, *rc_pos_r, *Ylm, *AO_sort, x2, y2, z2, x, y, z;
    
    // number of nodes in the local distributed domain
    DMnx = DMVertices[1] - DMVertices[0] + 1;
    DMny = DMVertices[3] - DMVertices[2] + 1;

    // (*AO_str) = (AO_OBJ *)malloc( sizeof(AO_OBJ) * pSPARC->Ntypes ); // TODO: deallocate!!
    double *Intgwt = NULL;
    double y0, z0, xi, yi, zi, ty, tz;
    double xin = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;
    if (pSPARC->CyclixFlag) {
        if(comm == pSPARC->kptcomm_topo){
            Intgwt = pSPARC->Intgwt_kpttopo;
        } else{
            Intgwt = pSPARC->Intgwt_psi;
        }
    }

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
        // allocate memory for projectors

        AO_str[ityp].Phi = (double **)malloc( sizeof(double *) * Atom_Influence_AO[ityp].n_atom);


        int n_AO = (pSPARC->AO_rad_str).num_orbitals[ityp];
        // psd_len = pSPARC->psd[ityp].size;
        psd_len = (pSPARC->AO_rad_str).N_rgrid[ityp];
        // number of projectors per atom
        AO_str[ityp].n_orbitals = 0;
        for (n_ao = 0; n_ao < n_AO; n_ao++) {
            l = (pSPARC->AO_rad_str).Rnl[ityp][n_ao].l;
            AO_str[ityp].n_orbitals += 2 * l + 1;
        }

        if (! AO_str[ityp].n_orbitals) continue;

        for (iat = 0; iat < Atom_Influence_AO[ityp].n_atom; iat++) {
            // store coordinates of the overlapping atom
            x0_i = Atom_Influence_AO[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_AO[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_AO[ityp].coords[iat*3+2];
            // grid nodes in (spherical) rc-domain
            ndc = Atom_Influence_AO[ityp].ndc[iat]; 
            AO_str[ityp].Phi[iat] = (double *)malloc( sizeof(double) * ndc * AO_str[ityp].n_orbitals); 

            rc_pos_x = (double *)malloc( sizeof(double) * ndc );
            rc_pos_y = (double *)malloc( sizeof(double) * ndc );
            rc_pos_z = (double *)malloc( sizeof(double) * ndc );
            rc_pos_r = (double *)malloc( sizeof(double) * ndc );
            Ylm = (double *)malloc( sizeof(double) * ndc );
            AO_sort = (double *)malloc( sizeof(double) * ndc );


            // use spline to fit UdV
            if(pSPARC->cell_typ == 0){
                for (ipos = 0; ipos < ndc; ipos++) {
                    indx = Atom_Influence_AO[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x2 = (i_DM + DMVertices[0]) * pSPARC->delta_x - x0_i;
                    y2 = (j_DM + DMVertices[2]) * pSPARC->delta_y - y0_i;
                    z2 = (k_DM + DMVertices[4]) * pSPARC->delta_z - z0_i;
                    rc_pos_x[ipos] = x2;
                    rc_pos_y[ipos] = y2;
                    rc_pos_z[ipos] = z2;
                    x2 *= x2; y2 *= y2; z2 *= z2;
                    rc_pos_r[ipos] = sqrt(x2+y2+z2);
                }
            } else if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
                for (ipos = 0; ipos < ndc; ipos++) {
                    indx = Atom_Influence_AO[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x = (i_DM + DMVertices[0]) * pSPARC->delta_x - x0_i;
                    y = (j_DM + DMVertices[2]) * pSPARC->delta_y - y0_i;
                    z = (k_DM + DMVertices[4]) * pSPARC->delta_z - z0_i;
                    nonCart2Cart_coord(pSPARC, &x, &y, &z);
                    rc_pos_x[ipos] = x;
                    rc_pos_y[ipos] = y;
                    rc_pos_z[ipos] = z;
                    x2 = x * x; y2 = y * y; z2 = z*z;
                    rc_pos_r[ipos] = sqrt(x2+y2+z2);
                }
            } 

            
            lcount = lcount2 = col_count = 0;
            // multiply spherical harmonics and UdV
            for (n_ao = 0; n_ao < (pSPARC->AO_rad_str).num_orbitals[ityp]; n_ao++) {
                l = (pSPARC->AO_rad_str).Rnl[ityp][n_ao].l;

                SplineInterpUniform((pSPARC->AO_rad_str).r_grid[ityp], (pSPARC->AO_rad_str).Rnl[ityp][n_ao].values, (pSPARC->AO_rad_str).Rnl[ityp][n_ao].num_count, 
                                            rc_pos_r, AO_sort, ndc, (pSPARC->AO_rad_str).SplineFitAO[ityp]+lcount*psd_len);
                for (int m = -l; m <= l; m++) {
                    RealSphericalHarmonic(ndc, rc_pos_x, rc_pos_y, rc_pos_z, rc_pos_r, l, m, Ylm);
                    
                    for (int ipos = 0; ipos < ndc; ipos++) {
                        AO_str[ityp].Phi[iat][col_count*ndc+ipos] = Ylm[ipos] * AO_sort[ipos];
                    }
                    col_count++;
                }
            }

            free(rc_pos_x);
            free(rc_pos_y);
            free(rc_pos_z);
            free(rc_pos_r);
            free(Ylm);
            free(AO_sort);
        }
    }

#ifdef DEBUG    
    if(!rank) printf(BLU "rank = %d, Time for spherical harmonics: %.3f ms\n" RESET, rank, t_tot*1e3);
#endif    

}


/**
 * @brief   Call Spline to calculate derivatives of the tabulated functions and
 *          store them for later use (during interpolation).
 */
void Calculate_SplineDerivAORadFun(SPARC_OBJ *pSPARC) {
    int ityp, l, lcount, lcount2, np, ppl_sum, psd_len;
    (pSPARC->AO_rad_str).SplineFitAO = (double **)malloc(sizeof(double*)*pSPARC->Ntypes);

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        int n_AO = (pSPARC->AO_rad_str).num_orbitals[ityp];
        psd_len = (pSPARC->AO_rad_str).N_rgrid[ityp];

        ppl_sum = 0;

        for (int n_ao = 0; n_ao < n_AO; n_ao++) {
            l = (pSPARC->AO_rad_str).Rnl[ityp][n_ao].l;
            ppl_sum += 2 * l + 1;
        }

        pSPARC->AO_rad_str.SplineFitAO[ityp] = (double *)malloc(sizeof(double)*psd_len * ppl_sum);
        if(pSPARC->AO_rad_str.SplineFitAO[ityp] == NULL) {
            printf("Memory allocation failed!\n");
            exit(EXIT_FAILURE);
        }
        lcount=0;
        for (int n_ao = 0; n_ao < n_AO; n_ao++) {
            l = (pSPARC->AO_rad_str).Rnl[ityp][n_ao].l;
            getYD_gen((pSPARC->AO_rad_str).r_grid[ityp], (pSPARC->AO_rad_str).Rnl[ityp][n_ao].values, (pSPARC->AO_rad_str).SplineFitAO[ityp]+lcount*psd_len, psd_len);
            lcount++;
        }
    }
}


/**
 * @brief   Calculate AtomicOrbitals. 
 */
void CalculateAtomicOrbitals_kpt(SPARC_OBJ *pSPARC, AO_OBJ *AO_str, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, int *DMVertices, MPI_Comm comm)
{
    // processors that are not in the dmcomm will continue
    if (comm == MPI_COMM_NULL) {
        return; // upon input, make sure process with bandcomm_index < 0 provides MPI_COMM_NULL
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    double t1, t2, t_tot, t_old;

    t_tot = t_old = 0.0;
#ifdef DEBUG
    if (rank == 0) printf("Calculate nonlocal projectors ... \n");
#endif    
    int l, n_ao, np, lcount, lcount2, m, psd_len, col_count, indx, ityp, iat, ipos, ndc, lloc, lmax, DMnx, DMny;
    int i_DM, j_DM, k_DM;
    double x0_i, y0_i, z0_i, *rc_pos_x, *rc_pos_y, *rc_pos_z, *rc_pos_r, *Ylm, *AO_sort, x2, y2, z2, x, y, z;
    
    // number of nodes in the local distributed domain
    DMnx = DMVertices[1] - DMVertices[0] + 1;
    DMny = DMVertices[3] - DMVertices[2] + 1;
    
    double *Intgwt = NULL;   
    double y0, z0, xi, yi, zi, ty, tz;
    double xin = pSPARC->xin + DMVertices[0] * pSPARC->delta_x; 
    if (pSPARC->CyclixFlag) {
        if(comm == pSPARC->kptcomm_topo){
            Intgwt = pSPARC->Intgwt_kpttopo;
        } else{
            Intgwt = pSPARC->Intgwt_psi;
        }
    }

    // (*AO_str) = (AO_OBJ *)malloc( sizeof(NLOC_PROJ_OBJ) * pSPARC->Ntypes ); // TODO: deallocate!!
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
        // allocate memory for projectors
        AO_str[ityp].Phi_c = (double _Complex **)malloc( sizeof(double _Complex *) * Atom_Influence_AO[ityp].n_atom);

        int n_AO = (pSPARC->AO_rad_str).num_orbitals[ityp];
        // psd_len = pSPARC->psd[ityp].size;
        psd_len = (pSPARC->AO_rad_str).N_rgrid[ityp];
        // number of projectors per atom
        AO_str[ityp].n_orbitals = 0;
        for (n_ao = 0; n_ao < n_AO; n_ao++) {
            l = (pSPARC->AO_rad_str).Rnl[ityp][n_ao].l;
            AO_str[ityp].n_orbitals += 2 * l + 1;
        }

        if (! AO_str[ityp].n_orbitals) continue;

        for (iat = 0; iat < Atom_Influence_AO[ityp].n_atom; iat++) {
            // store coordinates of the overlapping atom
            x0_i = Atom_Influence_AO[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_AO[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_AO[ityp].coords[iat*3+2];
            // grid nodes in (spherical) rc-domain
            ndc = Atom_Influence_AO[ityp].ndc[iat]; 
            AO_str[ityp].Phi_c[iat] = (double _Complex *)malloc( sizeof(double _Complex) * ndc * AO_str[ityp].n_orbitals); 
            rc_pos_x = (double *)malloc( sizeof(double) * ndc );
            rc_pos_y = (double *)malloc( sizeof(double) * ndc );
            rc_pos_z = (double *)malloc( sizeof(double) * ndc );
            rc_pos_r = (double *)malloc( sizeof(double) * ndc );
            Ylm = (double *)malloc( sizeof(double) * ndc );
            AO_sort = (double *)malloc( sizeof(double) * ndc );
            // use spline to fit UdV
            if(pSPARC->cell_typ == 0){
                for (ipos = 0; ipos < ndc; ipos++) {
                    indx = Atom_Influence_AO[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x2 = (i_DM + DMVertices[0]) * pSPARC->delta_x - x0_i;
                    y2 = (j_DM + DMVertices[2]) * pSPARC->delta_y - y0_i;
                    z2 = (k_DM + DMVertices[4]) * pSPARC->delta_z - z0_i;
                    rc_pos_x[ipos] = x2;
                    rc_pos_y[ipos] = y2;
                    rc_pos_z[ipos] = z2;
                    x2 *= x2; y2 *= y2; z2 *= z2;
                    rc_pos_r[ipos] = sqrt(x2+y2+z2);
                }
            } else if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
                for (ipos = 0; ipos < ndc; ipos++) {
                    indx = Atom_Influence_AO[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x = (i_DM + DMVertices[0]) * pSPARC->delta_x - x0_i;
                    y = (j_DM + DMVertices[2]) * pSPARC->delta_y - y0_i;
                    z = (k_DM + DMVertices[4]) * pSPARC->delta_z - z0_i;
                    nonCart2Cart_coord(pSPARC, &x, &y, &z);
                    rc_pos_x[ipos] = x;
                    rc_pos_y[ipos] = y;
                    rc_pos_z[ipos] = z;
                    x2 = x * x; y2 = y * y; z2 = z*z;
                    rc_pos_r[ipos] = sqrt(x2+y2+z2);
                }
            } 
            
            lcount = lcount2 = col_count = 0;
            // multiply spherical harmonics and UdV
            for (n_ao = 0; n_ao < (pSPARC->AO_rad_str).num_orbitals[ityp]; n_ao++) {
                l = (pSPARC->AO_rad_str).Rnl[ityp][n_ao].l;
                SplineInterpUniform((pSPARC->AO_rad_str).r_grid[ityp], (pSPARC->AO_rad_str).Rnl[ityp][n_ao].values, (pSPARC->AO_rad_str).Rnl[ityp][n_ao].num_count, 
                                            rc_pos_r, AO_sort, ndc, (pSPARC->AO_rad_str).SplineFitAO[ityp]+lcount*psd_len);
                                            
                for (int m = -l; m <= l; m++) {
                    RealSphericalHarmonic(ndc, rc_pos_x, rc_pos_y, rc_pos_z, rc_pos_r, l, m, Ylm);
                    // calculate Chi = UdV * Ylm
                    for (int ipos = 0; ipos < ndc; ipos++) {
                        AO_str[ityp].Phi_c[iat][col_count*ndc+ipos] = Ylm[ipos] * AO_sort[ipos];
                    }
                    col_count++;
                }
            }

            free(rc_pos_x);
            free(rc_pos_y);
            free(rc_pos_z);
            free(rc_pos_r);
            free(Ylm);
            free(AO_sort);

        }
    }
    
#ifdef DEBUG    
    if(!rank) printf(BLU"rank = %d, Time for spherical harmonics: %.3f ms\n"RESET, rank, t_tot*1e3);
#endif    
}

void Calculate_Overlap_AO(SPARC_OBJ *pSPARC, AO_OBJ *AO_str, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, double ******OIJ_local_images, int *DMVertices, MPI_Comm comm){
    // processors that are not in the dmcomm will continue
    if (comm == MPI_COMM_NULL) {
        return; // upon input, make sure process with bandcomm_index < 0 provides MPI_COMM_NULL
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2, t_tot, t_old;

    t_tot = t_old = 0.0;
#ifdef DEBUG
    if (rank == 0) printf("Calculate nonlocal projectors ... \n");
#endif    
    int l, np, lcount, lcount2, m, psd_len, col_count, indx, ityp, iat, ipos, ndc, lloc, lmax, DMnx, DMny;
    int i_DM, j_DM, k_DM;
    double x0_i, y0_i, z0_i, *rc_pos_x, *rc_pos_y, *rc_pos_z, *rc_pos_r, *Ylm, *AO_sort, x2, y2, z2, x, y, z;
    
    // number of nodes in the local distributed domain
    DMnx = DMVertices[1] - DMVertices[0] + 1;
    DMny = DMVertices[3] - DMVertices[2] + 1;

    double *Intgwt = NULL;
    double y0, z0, xi, yi, zi, ty, tz;
    double xin = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;
    if (pSPARC->CyclixFlag) {
        if(comm == pSPARC->kptcomm_topo){
            Intgwt = pSPARC->Intgwt_kpttopo;
        } else{
            Intgwt = pSPARC->Intgwt_psi;
        }
    }


    // double ******OIJ_local_images;
    OIJ_local_images = (double ******) malloc(sizeof(double*****)*pSPARC->Ntypes);
    for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
        OIJ_local_images[e1] = (double *****) malloc(sizeof(double****)*Atom_Influence_AO[e1].n_atom);
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e1];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }
        for (int i = 0; i < Atom_Influence_AO[e1].n_atom; i++){
            OIJ_local_images[e1][i] = (double ****) malloc(sizeof(double***)*n_orbital1);
            for (int ao1 = 0; ao1 < n_orbital1; ao1++){
                OIJ_local_images[e1][i][ao1] = (double ***) malloc(sizeof(double**)*pSPARC->Ntypes);
                for (int e2 = 0; e2 < pSPARC->Ntypes; e2++){
                    OIJ_local_images[e1][i][ao1][e2] = (double **) malloc(sizeof(double*)*Atom_Influence_AO[e2].n_atom);
                    int n_orbital2 = 0;
                    int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e2];
                    for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                        int l = (pSPARC->AO_rad_str).Rnl[e2][n_ao].l;
                        n_orbital2 += 2 * l + 1;
                    }
                    for (int j = 0; j < Atom_Influence_AO[e2].n_atom; j++){
                        OIJ_local_images[e1][i][ao1][e2][j] = (double *) calloc(n_orbital2, sizeof(double));
                    }
                }
            }
        }
    }

    int *ndc_idx_common1;
    int *ndc_idx_common2;
    int ndc_common;

    // int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *idx1, *idx2;
    int min_ndc;

    // double t1, t2;
    double *phi_local1, *phi_local2;
    for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e1];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }
        for (int i = 0; i < Atom_Influence_AO[e1].n_atom; i++){
            int ndc1 = Atom_Influence_AO[e1].ndc[i];
            idx1 = Atom_Influence_AO[e1].grid_pos[i];
            int xs1 = Atom_Influence_AO[e1].xs[i];
            int ys1 = Atom_Influence_AO[e1].ys[i];
            int zs1 = Atom_Influence_AO[e1].zs[i];

            int xe1 = Atom_Influence_AO[e1].xe[i];
            int ye1 = Atom_Influence_AO[e1].ye[i];
            int ze1 = Atom_Influence_AO[e1].ze[i];
            for (int e2 = 0; e2 < pSPARC->Ntypes; e2++){
                int n_orbital2 = 0;
                int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e2];
                for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                    int l = (pSPARC->AO_rad_str).Rnl[e2][n_ao].l;
                    n_orbital2 += 2 * l + 1;
                }
                for (int j = 0; j < Atom_Influence_AO[e2].n_atom; j++){
                    idx2 = Atom_Influence_AO[e2].grid_pos[j];
                    int xs2 = Atom_Influence_AO[e2].xs[j];
                    int ys2 = Atom_Influence_AO[e2].ys[j];
                    int zs2 = Atom_Influence_AO[e2].zs[j];

                    int xe2 = Atom_Influence_AO[e2].xe[j];
                    int ye2 = Atom_Influence_AO[e2].ye[j];
                    int ze2 = Atom_Influence_AO[e2].ze[j];

                    int xs, ys, zs, xe, ye, ze;
                    xs = max(xs1, xs2);
                    ys = max(ys1, ys2);
                    zs = max(zs1, zs2);
                    xe = min(xe1, xe2);
                    ye = min(ye1, ye2);
                    ze = min(ze1, ze2);

                    if ((xs>=xe)||(ys>=ye)||(zs>=ze)){
                        // This case is when there is no overlap
                        continue;
                    }
                    int ndc2 = Atom_Influence_AO[e2].ndc[j];

                    ndc_common = (xe-xs+1)*(ye-ys+1)*(ze-zs+1);
                    ndc_idx_common1 = (int *)malloc(sizeof(int)*ndc_common);
                    ndc_idx_common2 = (int *)malloc(sizeof(int)*ndc_common);
                    get_common_idx(Atom_Influence_AO[e1].grid_pos[i], Atom_Influence_AO[e2].grid_pos[j], ndc1, ndc2, &ndc_common, ndc_idx_common1, ndc_idx_common2);
                    if (ndc_common ==0){
                        continue;
                    }
                    t1 = MPI_Wtime();
                    for (int ao1 = 0; ao1 < n_orbital1; ao1++){
                        phi_local1 = &(AO_str[e1].Phi[i][ao1*ndc1]);
                        for (int ao2 = 0; ao2 < n_orbital2; ao2++){                            
                            phi_local2 = &(AO_str[e2].Phi[j][ao2*ndc2]);
                            for (int ndc_idx = 0; ndc_idx < ndc_common; ndc_idx++){
                                OIJ_local_images[e1][i][ao1][e2][j][ao2] += pSPARC->dV * phi_local1[ndc_idx_common1[ndc_idx]]*phi_local2[ndc_idx_common2[ndc_idx]];
                            }
                        }
                    }
                    free(ndc_idx_common1);
                    free(ndc_idx_common2);
                }
            }
            if (rank==0) printf("i: %d/%d\n",i, Atom_Influence_AO[e1].n_atom);
        }
    }





    double ****OIJ;
    OIJ = (double ****) malloc(sizeof(double***)*pSPARC->n_atom);
    int count1 = 0, count2 = 0;
    for (int e = 0; e < pSPARC->Ntypes; e++){
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }
        for (int i = 0; i < pSPARC->nAtomv[e]; i++){
            OIJ[i+count1] = (double ***) malloc(sizeof(double**)*n_orbital1);
            for (int j = 0; j < n_orbital1; j++){
                OIJ[i+count1][j] = (double **) malloc(sizeof(double*)*pSPARC->n_atom);
                count2 = 0;
                for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
                    int n_orbital2 = 0;
                    int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e1];
                    for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                        int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
                        n_orbital2 += 2 * l + 1;
                    }
                    for (int i1 = 0; i1 < pSPARC->nAtomv[e1]; i1++){
                        OIJ[i+count1][j][i1+count2] = (double *) calloc(n_orbital2, sizeof(double));
                    }   
                    count2 += pSPARC->nAtomv[e1];
                }
            }   
        }
        count1 += pSPARC->nAtomv[e];
    }



    int count_natom_e1 = 0, count_natom_e2 = 0;
    for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e1];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }

        for (int i = 0; i < Atom_Influence_AO[e1].n_atom; i++){
            int idx1 = Atom_Influence_AO[e1].atom_index[i];
            int na1 = idx1;

            for (int ao1 = 0; ao1 < n_orbital1; ao1++){
                count_natom_e2 = 0;
                for (int e2 = 0; e2 < pSPARC->Ntypes; e2++){
                    int n_orbital2 = 0;
                    int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e2];
                    for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                        int l = (pSPARC->AO_rad_str).Rnl[e2][n_ao].l;
                        n_orbital2 += 2 * l + 1;
                    }
                    for (int j = 0; j < Atom_Influence_AO[e2].n_atom; j++){
                        int idx2 = Atom_Influence_AO[e2].atom_index[j];
                        int na2 = idx2;
                        for (int ao2 = 0; ao2 < n_orbital2; ao2++){
                            OIJ[na1][ao1][na2][ao2] += OIJ_local_images[e1][i][ao1][e2][j][ao2];
                        }
                    }
                    count_natom_e2 += pSPARC->nAtomv[e2];
                }
            }
        }
        count_natom_e1 += pSPARC->nAtomv[e1];
    }

    
    
    int commsize;
    MPI_Comm_size(comm, &commsize);

    if (commsize > 1) {
        int count1 = 0, count2 = 0;
        for (int e = 0; e < pSPARC->Ntypes; e++){
            int n_orbital1 = 0;
            int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e];
            for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
                int l = (pSPARC->AO_rad_str).Rnl[e][n_ao].l;
                n_orbital1 += 2 * l + 1;
            }
            for (int i = 0; i < pSPARC->nAtomv[e]; i++){
                for (int j = 0; j < n_orbital1; j++){
                    count2 = 0;
                    for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
                        int n_orbital2 = 0;
                        int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e1];
                        for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                            int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
                            n_orbital2 += 2 * l + 1;
                        }
                        for (int i1 = 0; i1 < pSPARC->nAtomv[e1]; i1++){
                            for (int j1 = 0; j1 < n_orbital2; j1++){
                                MPI_Allreduce(MPI_IN_PLACE, &OIJ[i+count1][j][i1+count2][j1], 1, MPI_DOUBLE, MPI_SUM, comm);
                            }
                        }
                        count2 += pSPARC->nAtomv[e1];
                    }
                }   
            }
            count1 += pSPARC->nAtomv[e];
        }
    }


    int rank_comm, count3;
    MPI_Comm_rank(comm, &rank_comm);
    char fname[] = "Overlap_integrals";
    char result[1000];

    int tot_orb = 0;
    for (int e = 0; e < pSPARC->Ntypes; e++){
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }
        tot_orb +=  n_orbital1 * pSPARC->nAtomv[e];
    }

    double *Oij_mat;


    FILE *fp;
    if (rank==0){
        Oij_mat = (double *) malloc(sizeof(double) * tot_orb*tot_orb);
        snprintf(result, sizeof(result), "%s_%s_kpt_gamma.txt", pSPARC->filename_out, fname);
        fp = fopen(result,"w");
        fprintf(fp, "kpt: %f %f %f\n", 0.0, 0.0, 0.0);

        count1 = 0, count2 = 0, count3 = 0;
         
        for (int e = 0; e < pSPARC->Ntypes; e++){
            int n_orbital1 = 0;
            int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e];
            for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
                int l = (pSPARC->AO_rad_str).Rnl[e][n_ao].l;
                n_orbital1 += 2 * l + 1;
            }
            for (int i = 0; i < pSPARC->nAtomv[e]; i++){
                for (int j = 0; j < n_orbital1; j++){
                    count2 = 0;
                    for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
                        int n_orbital2 = 0;
                        int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e1];
                        for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                            int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
                            n_orbital2 += 2 * l + 1;
                        }
                        for (int i1 = 0; i1 < pSPARC->nAtomv[e1]; i1++){
                            for (int j1 = 0; j1 < n_orbital2; j1++){
                                Oij_mat[count3] = OIJ[count1+i][j][count2+i1][j1];
                                count3++;
                            }
                        }
                        count2 += pSPARC->nAtomv[e1];
                    }
                }   
            }
            count1 += pSPARC->nAtomv[e];
        }

        count3 = 0;
        for (int i = 0; i < tot_orb; i++){
            for (int j = 0; j < tot_orb; j++){
                fprintf(fp, "%.6E ", Oij_mat[count3]);
                count3++;
            }
            fprintf(fp,"\n");
        }
        fclose(fp);
        free(Oij_mat);
    }


    for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e1];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }
        for (int i = 0; i < Atom_Influence_AO[e1].n_atom; i++){
            for (int ao1 = 0; ao1 < n_orbital1; ao1++){
                for (int e2 = 0; e2 < pSPARC->Ntypes; e2++){
                    int n_orbital2 = 0;
                    int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e2];
                    for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                        int l = (pSPARC->AO_rad_str).Rnl[e2][n_ao].l;
                        n_orbital2 += 2 * l + 1;
                    }
                    for (int j = 0; j < Atom_Influence_AO[e2].n_atom; j++){
                        free(OIJ_local_images[e1][i][ao1][e2][j]);
                    }
                    free(OIJ_local_images[e1][i][ao1][e2]);
                }
                free(OIJ_local_images[e1][i][ao1]);
            }
            free(OIJ_local_images[e1][i]);
        }
        free(OIJ_local_images[e1]);
    }
    free(OIJ_local_images);

    count1 = 0, count2 = 0;
    for (int e = 0; e < pSPARC->Ntypes; e++){
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }
        for (int i = 0; i < pSPARC->nAtomv[e]; i++){
            for (int j = 0; j < n_orbital1; j++){
                count2 = 0;
                for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
                    int n_orbital2 = 0;
                    int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e1];
                    for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                        int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
                        n_orbital2 += 2 * l + 1;
                    }
                    for (int i1 = 0; i1 < pSPARC->nAtomv[e1]; i1++){
                        free(OIJ[i+count1][j][i1+count2]);
                    }   
                    count2 += pSPARC->nAtomv[e1];
                }
                free(OIJ[i+count1][j]);
            }   
            free(OIJ[i+count1]);
        }
        count1 += pSPARC->nAtomv[e];
    }
    free(OIJ);

}

void Calculate_Overlap_AO_kpt(SPARC_OBJ *pSPARC, AO_OBJ *AO_str, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, int kpt, double _Complex ******OIJ_local_images, int *DMVertices, MPI_Comm comm){
    // processors that are not in the dmcomm will continue
    if (comm == MPI_COMM_NULL) {
        return; // upon input, make sure process with bandcomm_index < 0 provides MPI_COMM_NULL
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2, t_tot, t_old;

    t_tot = t_old = 0.0;
#ifdef DEBUG
    if (rank == 0) printf("Calculate nonlocal projectors ... \n");
#endif    
    int l, np, lcount, lcount2, m, psd_len, col_count, indx, ityp, iat, ipos, ndc, lloc, lmax, DMnx, DMny;
    int i_DM, j_DM, k_DM;
    double x0_i, y0_i, z0_i, *rc_pos_x, *rc_pos_y, *rc_pos_z, *rc_pos_r, *Ylm, *AO_sort, x2, y2, z2, x, y, z;
    
    // number of nodes in the local distributed domain
    DMnx = DMVertices[1] - DMVertices[0] + 1;
    DMny = DMVertices[3] - DMVertices[2] + 1;

    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1 = pSPARC->k1_loc[kpt];
    double k2 = pSPARC->k2_loc[kpt];
    double k3 = pSPARC->k3_loc[kpt];

    double theta;
    double _Complex bloch_fac1, bloch_fac2;


    double *Intgwt = NULL;
    double y0, z0, xi, yi, zi, ty, tz;
    double xin = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;
    if (pSPARC->CyclixFlag) {
        if(comm == pSPARC->kptcomm_topo){
            Intgwt = pSPARC->Intgwt_kpttopo;
        } else{
            Intgwt = pSPARC->Intgwt_psi;
        }
    }


    // double ******OIJ_local_images;
    OIJ_local_images = (double _Complex ******) malloc(sizeof(double _Complex*****)*pSPARC->Ntypes);
    for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
        OIJ_local_images[e1] = (double _Complex *****) malloc(sizeof(double _Complex****)*Atom_Influence_AO[e1].n_atom);
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e1];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }
        for (int i = 0; i < Atom_Influence_AO[e1].n_atom; i++){
            OIJ_local_images[e1][i] = (double _Complex ****) malloc(sizeof(double _Complex***)*n_orbital1);
            for (int ao1 = 0; ao1 < n_orbital1; ao1++){
                OIJ_local_images[e1][i][ao1] = (double _Complex ***) malloc(sizeof(double _Complex**)*pSPARC->Ntypes);
                for (int e2 = 0; e2 < pSPARC->Ntypes; e2++){
                    OIJ_local_images[e1][i][ao1][e2] = (double _Complex **) malloc(sizeof(double _Complex*)*Atom_Influence_AO[e2].n_atom);
                    int n_orbital2 = 0;
                    int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e2];
                    for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                        int l = (pSPARC->AO_rad_str).Rnl[e2][n_ao].l;
                        n_orbital2 += 2 * l + 1;
                    }
                    for (int j = 0; j < Atom_Influence_AO[e2].n_atom; j++){
                        OIJ_local_images[e1][i][ao1][e2][j] = (double _Complex *) calloc(n_orbital2, sizeof(double _Complex));
                    }
                }
            }
        }
    }

    int *ndc_idx_common1;
    int *ndc_idx_common2;
    int ndc_common;

    
    double _Complex *phi_local1, *phi_local2;
    for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e1];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }
        for (int i = 0; i < Atom_Influence_AO[e1].n_atom; i++){
            x0_i = Atom_Influence_AO[e1].coords[i*3  ];
            y0_i = Atom_Influence_AO[e1].coords[i*3+1];
            z0_i = Atom_Influence_AO[e1].coords[i*3+2];

            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac1 = cos(theta) + sin(theta) * I;


            int xs1 = Atom_Influence_AO[e1].xs[i];
            int ys1 = Atom_Influence_AO[e1].ys[i];
            int zs1 = Atom_Influence_AO[e1].zs[i];

            int xe1 = Atom_Influence_AO[e1].xe[i];
            int ye1 = Atom_Influence_AO[e1].ye[i];
            int ze1 = Atom_Influence_AO[e1].ze[i];
            for (int ao1 = 0; ao1 < n_orbital1; ao1++){
                
                int ndc1 = Atom_Influence_AO[e1].ndc[i];
                // phi_local1 = (double *)malloc(sizeof(double)*ndc1);
                // fill phi_local1
                phi_local1 = &(AO_str[e1].Phi_c[i][ao1*ndc1]);



                for (int e2 = 0; e2 < pSPARC->Ntypes; e2++){
                    int n_orbital2 = 0;
                    int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e2];
                    for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                        int l = (pSPARC->AO_rad_str).Rnl[e2][n_ao].l;
                        n_orbital2 += 2 * l + 1;
                    }
                    for (int j = 0; j < Atom_Influence_AO[e2].n_atom; j++){
                        int xs2 = Atom_Influence_AO[e2].xs[j];
                        int ys2 = Atom_Influence_AO[e2].ys[j];
                        int zs2 = Atom_Influence_AO[e2].zs[j];

                        int xe2 = Atom_Influence_AO[e2].xe[j];
                        int ye2 = Atom_Influence_AO[e2].ye[j];
                        int ze2 = Atom_Influence_AO[e2].ze[j];

                        x0_i = Atom_Influence_AO[e2].coords[j*3  ];
                        y0_i = Atom_Influence_AO[e2].coords[j*3+1];
                        z0_i = Atom_Influence_AO[e2].coords[j*3+2];

                        theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
                        bloch_fac2 = cos(theta) - sin(theta) * I;


                        int xs, ys, zs, xe, ye, ze;
                        xs = max(xs1, xs2);
                        ys = max(ys1, ys2);
                        zs = max(zs1, zs2);
                        xe = min(xe1, xe2);
                        ye = min(ye1, ye2);
                        ze = min(ze1, ze2);

                        if ((xs>=xe)||(ys>=ye)||(zs>=ze)){
                            // This case is when there is no overlap
                            continue;
                        }


                        int ndc2 = Atom_Influence_AO[e2].ndc[j];

                        ndc_common = (xe-xs+1)*(ye-ys+1)*(ze-zs+1);
                        ndc_idx_common1 = (int *)malloc(sizeof(int)*ndc_common);
                        ndc_idx_common2 = (int *)malloc(sizeof(int)*ndc_common);

                        get_common_idx(Atom_Influence_AO[e1].grid_pos[i], Atom_Influence_AO[e2].grid_pos[j], ndc1, ndc2, &ndc_common, ndc_idx_common1, ndc_idx_common2);
                        if (ndc_common ==0){
                            continue;
                        }

                        for (int ao2 = 0; ao2 < n_orbital2; ao2++){                            
                            phi_local2 = &(AO_str[e2].Phi_c[j][ao2*ndc2]);
                            for (int ndc_idx = 0; ndc_idx < ndc_common; ndc_idx++){
                                OIJ_local_images[e1][i][ao1][e2][j][ao2] += pSPARC->dV * bloch_fac1 * bloch_fac2 * phi_local1[ndc_idx_common1[ndc_idx]] * phi_local2[ndc_idx_common2[ndc_idx]];
                            }
                             
                        }

                        free(ndc_idx_common1);
                        free(ndc_idx_common2);
                    }

                }
            }
            if (rank==0) printf("i: %d/%d\n",i, Atom_Influence_AO[e1].n_atom);
        }
    }


    double _Complex ****OIJ;
    OIJ = (double _Complex ****) malloc(sizeof(double _Complex***)*pSPARC->n_atom);
    int count1 = 0, count2 = 0;
    for (int e = 0; e < pSPARC->Ntypes; e++){
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }
        for (int i = 0; i < pSPARC->nAtomv[e]; i++){
            OIJ[i+count1] = (double _Complex ***) malloc(sizeof(double _Complex**)*n_orbital1);
            for (int j = 0; j < n_orbital1; j++){
                OIJ[i+count1][j] = (double _Complex **) malloc(sizeof(double _Complex*)*pSPARC->n_atom);
                count2 = 0;
                for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
                    int n_orbital2 = 0;
                    int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e1];
                    for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                        int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
                        n_orbital2 += 2 * l + 1;
                    }
                    for (int i1 = 0; i1 < pSPARC->nAtomv[e1]; i1++){
                        OIJ[i+count1][j][i1+count2] = (double _Complex *) calloc(n_orbital2, sizeof(double _Complex));
                    }   
                    count2 += pSPARC->nAtomv[e1];
                }
            }   
        }
        count1 += pSPARC->nAtomv[e];
    }


    int count_natom_e1 = 0, count_natom_e2 = 0;
    for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e1];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }

        for (int i = 0; i < Atom_Influence_AO[e1].n_atom; i++){
            int idx1 = Atom_Influence_AO[e1].atom_index[i];
            int na1 = idx1;

            for (int ao1 = 0; ao1 < n_orbital1; ao1++){
                count_natom_e2 = 0;
                for (int e2 = 0; e2 < pSPARC->Ntypes; e2++){
                    int n_orbital2 = 0;
                    int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e2];
                    for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                        int l = (pSPARC->AO_rad_str).Rnl[e2][n_ao].l;
                        n_orbital2 += 2 * l + 1;
                    }
                    for (int j = 0; j < Atom_Influence_AO[e2].n_atom; j++){
                        int idx2 = Atom_Influence_AO[e2].atom_index[j];
                        int na2 = idx2;
                        for (int ao2 = 0; ao2 < n_orbital2; ao2++){
                            OIJ[na1][ao1][na2][ao2] += OIJ_local_images[e1][i][ao1][e2][j][ao2];
                        }
                    }
                    count_natom_e2 += pSPARC->nAtomv[e2];
                }
            }
        }
        count_natom_e1 += pSPARC->nAtomv[e1];
    }

    
    
    int commsize;
    MPI_Comm_size(comm, &commsize);

    if (commsize > 1) {
        int count1 = 0, count2 = 0;
        for (int e = 0; e < pSPARC->Ntypes; e++){
            int n_orbital1 = 0;
            int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e];
            for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
                int l = (pSPARC->AO_rad_str).Rnl[e][n_ao].l;
                n_orbital1 += 2 * l + 1;
            }
            for (int i = 0; i < pSPARC->nAtomv[e]; i++){
                for (int j = 0; j < n_orbital1; j++){
                    count2 = 0;
                    for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
                        int n_orbital2 = 0;
                        int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e1];
                        for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                            int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
                            n_orbital2 += 2 * l + 1;
                        }
                        for (int i1 = 0; i1 < pSPARC->nAtomv[e1]; i1++){
                            for (int j1 = 0; j1 < n_orbital2; j1++){
                                MPI_Allreduce(MPI_IN_PLACE, &OIJ[i+count1][j][i1+count2][j1], 1, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
                            }
                        }
                        count2 += pSPARC->nAtomv[e1];
                    }
                }   
            }
            count1 += pSPARC->nAtomv[e];
        }
    }
    

    int rank_comm, count3;
    MPI_Comm_rank(comm, &rank_comm);
    char fname[] = "Overlap_integrals";
    char result[1000];

    int tot_orb = 0;
    for (int e = 0; e < pSPARC->Ntypes; e++){
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }
        tot_orb +=  n_orbital1 * pSPARC->nAtomv[e];
    }

    double _Complex *Oij_mat;


    FILE *fp;
    if (rank_comm==0){
        printf("tot_orb: %d\n", tot_orb);
        Oij_mat = (double _Complex *) malloc(sizeof(double _Complex) * tot_orb*tot_orb);
        snprintf(result, sizeof(result), "%s_%s_kpt_%d.txt", pSPARC->filename_out, fname, pSPARC->kpt_start_indx+kpt);
        fp = fopen(result,"w");
        fprintf(fp, "kpt: %f %f %f\n", k1, k2,k3);

        count1 = 0, count2 = 0, count3 = 0;
         
        for (int e = 0; e < pSPARC->Ntypes; e++){
            int n_orbital1 = 0;
            int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e];
            for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
                int l = (pSPARC->AO_rad_str).Rnl[e][n_ao].l;
                n_orbital1 += 2 * l + 1;
            }
            for (int i = 0; i < pSPARC->nAtomv[e]; i++){
                for (int j = 0; j < n_orbital1; j++){
                    count2 = 0;
                    for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
                        int n_orbital2 = 0;
                        int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e1];
                        for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                            int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
                            n_orbital2 += 2 * l + 1;
                        }
                        for (int i1 = 0; i1 < pSPARC->nAtomv[e1]; i1++){
                            for (int j1 = 0; j1 < n_orbital2; j1++){
                                Oij_mat[count3] = OIJ[count1][j][count2][j1];
                                count3++;
                            }
                        }
                        count2 += pSPARC->nAtomv[e1];
                    }
                }   
            }
            count1 += pSPARC->nAtomv[e];
        }

        count3 = 0;
        for (int i = 0; i < tot_orb; i++){
            for (int j = 0; j < tot_orb; j++){
                fprintf(fp, "%.6E ", creal(Oij_mat[count3]));
                count3++;
            }
            fprintf(fp,"\n");
        }

        fprintf(fp,"\n");
        count3 = 0;
        for (int i = 0; i < tot_orb; i++){
            for (int j = 0; j < tot_orb; j++){
                fprintf(fp, "%.6E ", cimag(Oij_mat[count3]));
                count3++;
            }
            fprintf(fp,"\n");
        }

        fclose(fp);
        free(Oij_mat);
    }


    for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e1];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }
        for (int i = 0; i < Atom_Influence_AO[e1].n_atom; i++){
            for (int ao1 = 0; ao1 < n_orbital1; ao1++){
                for (int e2 = 0; e2 < pSPARC->Ntypes; e2++){
                    int n_orbital2 = 0;
                    int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e2];
                    for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                        int l = (pSPARC->AO_rad_str).Rnl[e2][n_ao].l;
                        n_orbital2 += 2 * l + 1;
                    }
                    for (int j = 0; j < Atom_Influence_AO[e2].n_atom; j++){
                        free(OIJ_local_images[e1][i][ao1][e2][j]);
                    }
                    free(OIJ_local_images[e1][i][ao1][e2]);
                }
                free(OIJ_local_images[e1][i][ao1]);
            }
            free(OIJ_local_images[e1][i]);
        }
        free(OIJ_local_images[e1]);
    }
    free(OIJ_local_images);



    count1 = 0, count2 = 0;
    for (int e = 0; e < pSPARC->Ntypes; e++){
        int n_orbital1 = 0;
        int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[e];
        for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[e][n_ao].l;
            n_orbital1 += 2 * l + 1;
        }
        for (int i = 0; i < pSPARC->nAtomv[e]; i++){
            for (int j = 0; j < n_orbital1; j++){
                count2 = 0;
                for (int e1 = 0; e1 < pSPARC->Ntypes; e1++){
                    int n_orbital2 = 0;
                    int n_AO2 = (pSPARC->AO_rad_str).num_orbitals[e1];
                    for (int n_ao = 0; n_ao < n_AO2; n_ao++) {
                        int l = (pSPARC->AO_rad_str).Rnl[e1][n_ao].l;
                        n_orbital2 += 2 * l + 1;
                    }
                    for (int i1 = 0; i1 < pSPARC->nAtomv[e1]; i1++){
                        free(OIJ[i+count1][j][i1+count2]);
                    }   
                    count2 += pSPARC->nAtomv[e1];
                }
                free(OIJ[i+count1][j]);
            }   
            free(OIJ[i+count1]);
        }
        count1 += pSPARC->nAtomv[e];
    }
    free(OIJ);


}


void get_common_idx(int *x, int *y, int nx, int ny, int *n_common, int *nx_common, int *ny_common) {
    int i = 0;    // Index for array x
    int j = 0;    // Index for array y
    int count = 0;  // Counter for common elements

    // Compare elements using two pointers since arrays are sorted
    while (i < nx && j < ny) {
        int xi = x[i];
        int yj = y[j];
        if (xi < yj) {
            i++;
        } else if (xi > yj) {
            j++;
        } else {  // Found a common element
            nx_common[count] = i;
            ny_common[count] = j;
            count++;
            i++;
            j++;
        }
    }
    *n_common = count;
}




/**
 * @brief   Calculate indices for storing nonlocal inner product in an array. 
 *
 *          We will store the inner product < Chi_Jlm, x_n > in a continuous array "alpha",
 *          the dimensions are in the order: <lm>, n, J. Here we find out the sizes of the 
 *          inner product corresponding to atom J, and the total number of inner products
 *          corresponding to each vector x_n.
 */
void CalculateAOInnerProductIndex(SPARC_OBJ *pSPARC)
{
    int ityp, iat, l, atom_index, n_orbital;

    (pSPARC->AO_rad_str).IP_displ = (int *)malloc( sizeof(int) * (pSPARC->n_atom+1));
    
    atom_index = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        int n_AO = (pSPARC->AO_rad_str).num_orbitals[ityp];
        // number of projectors per atom
        n_orbital = 0;
        for (int n_ao = 0; n_ao < n_AO; n_ao++) {
            int l = (pSPARC->AO_rad_str).Rnl[ityp][n_ao].l;
            n_orbital += 2 * l + 1;
        }


        (pSPARC->AO_rad_str).IP_displ[0] = 0;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            (pSPARC->AO_rad_str).IP_displ[atom_index+1] = (pSPARC->AO_rad_str).IP_displ[atom_index] + n_orbital;

            atom_index++;
        }
    }
}



/**
 * @brief   Calculate Vnl times vectors in a matrix-free way.
 */
void AO_psi_mult(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, 
                  AO_OBJ *AO_str, int ncol, double *x, int ldi, double *Hx, int i_spin, int ldo, MPI_Comm comm)
{   
    int rank_comm;
    // processors that are not in the dmcomm will continue
    if (comm == MPI_COMM_NULL) {
        // return; // upon input, make sure process with bandcomm_index < 0 provides MPI_COMM_NULL
        rank_comm = -1;
    } else {
        MPI_Comm_rank(comm, &rank_comm);
    }


    int i, n, np, count;
    /* compute nonlocal operator times vector(s) */
    int ityp, iat, l, m, ldispl, lmax, ndc, atom_index;
    double *alpha, *x_rc, *Vnlx;
    alpha = (double *)calloc( (pSPARC->AO_rad_str).IP_displ[pSPARC->n_atom] * ncol, sizeof(double));
    //first find inner product

    if (comm != MPI_COMM_NULL){
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            int n_orbital1 = 0;
            int n_AO1 = (pSPARC->AO_rad_str).num_orbitals[ityp];
            for (int n_ao = 0; n_ao < n_AO1; n_ao++) {
                int l = (pSPARC->AO_rad_str).Rnl[ityp][n_ao].l;
                n_orbital1 += 2 * l + 1;
            }

            if (! n_orbital1) continue; // this is typical for hydrogen
            for (iat = 0; iat < Atom_Influence_AO[ityp].n_atom; iat++) {
                ndc = Atom_Influence_AO[ityp].ndc[iat]; 
                x_rc = (double *)malloc( ndc * ncol * sizeof(double));
                atom_index = Atom_Influence_AO[ityp].atom_index[iat];
                for (n = 0; n < ncol; n++) {
                    for (i = 0; i < ndc; i++) {
                        x_rc[n*ndc+i] = x[n*ldi + Atom_Influence_AO[ityp].grid_pos[iat][i]];
                    }
                }
                if (pSPARC->CyclixFlag) {
                    // cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, nlocProj[ityp].nproj, ncol, ndc, 
                    //     1.0, nlocProj[ityp].Chi_cyclix[iat], ndc, x_rc, ndc, 1.0, 
                    //     alpha+pSPARC->IP_displ[atom_index]*ncol, nlocProj[ityp].nproj);
                } else {
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n_orbital1, ncol, ndc, 
                        sqrt(pSPARC->dV), AO_str[ityp].Phi[iat], ndc, x_rc, ndc, 1.0, 
                        alpha+(pSPARC->AO_rad_str).IP_displ[atom_index]*ncol, n_orbital1);          
                }
                free(x_rc);
            }
        }
    }
    




    // if there are domain parallelization over each band, we need to sum over all processes over domain comm
    int commsize;
     if (comm != MPI_COMM_NULL){
        MPI_Comm_size(comm, &commsize);
        if (commsize > 1) {
            MPI_Allreduce(MPI_IN_PLACE, alpha, (pSPARC->AO_rad_str).IP_displ[pSPARC->n_atom] * ncol, MPI_DOUBLE, MPI_SUM, comm);
        }
     }
    


    FILE *fp;
    // int rank_comm;
    int count3;
   
    char fname[] = "Projections";
    char result[1000];


    int *recv_counts = NULL;
    int *displs = NULL;
    double *gathered_alpha;
    int rank, size;
    int rank1;
    int number  = rank_comm;
    int root = 0;
    
    int size_alpha = ncol*(pSPARC->AO_rad_str).IP_displ[pSPARC->n_atom];
    int num_zeros = pSPARC->npband; 
    int total_size;

    MPI_Comm_rank(pSPARC->kptcomm, &rank);
    MPI_Comm_size(pSPARC->kptcomm, &size);

    // printf("rank: %d, size: %d\n", rank, size);
    // return;

    int color = (number == 0) ? 1 : MPI_UNDEFINED;
    MPI_Comm new_comm;
    MPI_Comm_split(pSPARC->kptcomm, color, rank, &new_comm);
    

    int size_newcomm;
    if (number==0){
        MPI_Comm_rank(new_comm, &rank1);
    }
    
    

    if (number == 0) {
        if (rank1 == root) {
            recv_counts = (int *)malloc(num_zeros * sizeof(int));
            displs = (int *)malloc(num_zeros * sizeof(int));
        }
        // Gather sizes from ranks with number == 0
        MPI_Gather(&size_alpha, 1, MPI_INT, recv_counts, 1, MPI_INT, root, new_comm);
        // Root computes displacements
        if (rank1 == root) {
            total_size = 0;
            for (int i = 0; i < num_zeros; i++) {
                displs[i] = total_size;
                total_size += recv_counts[i];
            }
            gathered_alpha = (double *)malloc(total_size * sizeof(double));
        }
    }

    if (number == 0) {
        MPI_Gatherv(alpha, size_alpha, MPI_DOUBLE,
                    gathered_alpha, recv_counts, displs, MPI_DOUBLE,
                    root, new_comm);
    }
    
    int ncol_total = 0;

    if (rank_comm==0 && rank1==root){
        snprintf(result, sizeof(result), "%s_%s_spin_%d_kpt_gamma.txt", pSPARC->filename_out, fname, i_spin);
        fp = fopen(result,"w");
        fprintf(fp, "kpt: %f %f %f, spin_i: %d\n", 0.0, 0.0, 0.0, i_spin);
        count3=0;
        ncol_total = total_size/(pSPARC->AO_rad_str).IP_displ[pSPARC->n_atom];
        for (int i = 0; i < ncol_total; i++){
            for (int j =0; j < (pSPARC->AO_rad_str).IP_displ[pSPARC->n_atom]; j++){
                fprintf(fp, "%.6E ", gathered_alpha[count3]);
                count3++;
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        free(gathered_alpha);
    }

    if (number == 0){
        MPI_Comm_free(&new_comm);
    }
    

    free(alpha);


}


/**
 * @brief   Calculate Vnl times vectors in a matrix-free way with Bloch factor
 */
void AO_psi_mult_kpt(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_AO, 
                      AO_OBJ *AO_str, int ncol, double _Complex *x, int ldi, double _Complex *Hx, int i_spin, int ldo, int kpt, MPI_Comm comm)
{   
    int rank_comm;
    // processors that are not in the dmcomm will continue
    if (comm == MPI_COMM_NULL) {
        // return; // upon input, make sure process with bandcomm_index < 0 provides MPI_COMM_NULL
        rank_comm = -1;
    } else {
        MPI_Comm_rank(comm, &rank_comm);
    }

    int i, n, np, count;
    /* compute nonlocal operator times vector(s) */
    int ityp, iat, l, m, ldispl, lmax, ndc, atom_index;
    double x0_i, y0_i, z0_i;
    double _Complex *alpha, *x_rc, *Vnlx;
    alpha = (double _Complex *) calloc( (pSPARC->AO_rad_str).IP_displ[pSPARC->n_atom] * ncol, sizeof(double _Complex));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1 = pSPARC->k1_loc[kpt];
    double k2 = pSPARC->k2_loc[kpt];
    double k3 = pSPARC->k3_loc[kpt];
    double theta;
    double _Complex bloch_fac, a, b;

    FILE *fpx;
    //first find inner product
    if (comm != MPI_COMM_NULL){
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (! AO_str[ityp].n_orbitals) continue; // this is typical for hydrogen
            for (iat = 0; iat < Atom_Influence_AO[ityp].n_atom; iat++) {
                x0_i = Atom_Influence_AO[ityp].coords[iat*3  ];
                y0_i = Atom_Influence_AO[ityp].coords[iat*3+1];
                z0_i = Atom_Influence_AO[ityp].coords[iat*3+2];
                theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
                bloch_fac = cos(theta) + sin(theta) * I;
                if (pSPARC->CyclixFlag) {
                    a = bloch_fac;
                } else {
                    a = bloch_fac * pSPARC->dV;
                }
                b = 1.0;
                ndc = Atom_Influence_AO[ityp].ndc[iat]; 
                x_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
                atom_index = Atom_Influence_AO[ityp].atom_index[iat];
                for (n = 0; n < ncol; n++) {
                    for (i = 0; i < ndc; i++) {
                        x_rc[n*ndc+i] = x[n*ldi + Atom_Influence_AO[ityp].grid_pos[iat][i]];
                    }
                }
                if (pSPARC->CyclixFlag) {
                    // cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nlocProj[ityp].nproj, ncol, ndc, 
                    //     &a, nlocProj[ityp].Chi_c_cyclix[iat], ndc, x_rc, ndc, &b, 
                    //     alpha+pSPARC->IP_displ[atom_index]*ncol, nlocProj[ityp].nproj);
                    // Do nothing
                } else {
                    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, AO_str[ityp].n_orbitals, ncol, ndc, 
                        &a, AO_str[ityp].Phi_c[iat], ndc, x_rc, ndc, &b, 
                        alpha+(pSPARC->AO_rad_str).IP_displ[atom_index]*ncol, AO_str[ityp].n_orbitals);
                }
                free(x_rc);
            }
        }
    }
    


    // if there are domain parallelization over each band, we need to sum over all processes over domain comm
    int commsize;
    if (comm != MPI_COMM_NULL){
        MPI_Comm_size(comm, &commsize);
        if (commsize > 1) {
            MPI_Allreduce(MPI_IN_PLACE, alpha, (pSPARC->AO_rad_str).IP_displ[pSPARC->n_atom] * ncol, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
        }
    }
    

        

    // int commsize;
    // MPI_Comm_size(comm, &commsize);
    // if (commsize > 1) {
    //     MPI_Allreduce(MPI_IN_PLACE, alpha, (pSPARC->AO_rad_str).IP_displ[pSPARC->n_atom] * ncol, MPI_DOUBLE, MPI_SUM, comm);
    // }

    FILE *fp;
    int count3;
    MPI_Comm_rank(comm, &rank_comm);
    char fname[] = "Projections";
    char result[1000];

    int *recv_counts = NULL;
    int *displs = NULL;
    double _Complex *gathered_alpha;
    int rank, rank1, size;
    int number  = rank_comm;
    int root = 0;
    
    int size_alpha = ncol*(pSPARC->AO_rad_str).IP_displ[pSPARC->n_atom];
    int num_zeros = pSPARC->npband; 
    int total_size;

    MPI_Comm_rank(pSPARC->kptcomm, &rank);
    MPI_Comm_size(pSPARC->kptcomm, &size);


    // MPI_Comm new_comm;
    // MPI_Comm_split(pSPARC->kptcomm, number == 0, rank, &new_comm);
    // MPI_Comm_rank(new_comm, &rank1);


    int color = (number == 0) ? 1 : MPI_UNDEFINED;
    MPI_Comm new_comm;
    MPI_Comm_split(pSPARC->kptcomm, color, rank, &new_comm);
    
    if (number==0){
        MPI_Comm_rank(new_comm, &rank1);
    }

    if (number == 0) {
        if (rank1 == root) {
            recv_counts = (int *)malloc(num_zeros * sizeof(int));
            displs = (int *)malloc(num_zeros * sizeof(int));
        }
        // Gather sizes from ranks with number == 0
        MPI_Gather(&size_alpha, 1, MPI_INT, recv_counts, 1, MPI_INT, root, new_comm);
        // Root computes displacements
        if (rank1 == root) {
            total_size = 0;
            for (int i = 0; i < num_zeros; i++) {
                displs[i] = total_size;
                total_size += recv_counts[i];
            }
            gathered_alpha = (double _Complex *)malloc(total_size * sizeof(double _Complex));
        }
    }




    if (number == 0) {
        MPI_Gatherv(alpha, size_alpha, MPI_DOUBLE_COMPLEX,
                    gathered_alpha, recv_counts, displs, MPI_DOUBLE_COMPLEX,
                    root, new_comm);
    }
    

    int ncol_total = 0;
    if (rank_comm==0 && rank1==root){
        snprintf(result, sizeof(result), "%s_%s_spin_%d_kpt_%d.txt", pSPARC->filename_out, fname, pSPARC->spin_start_indx+i_spin, pSPARC->kpt_start_indx+kpt);
        fp = fopen(result,"w");
        fprintf(fp, "kpt: %f %f %f, spin_i: %d\n", k1, k2, k3, i_spin);
        count3=0;
        ncol_total = total_size/(pSPARC->AO_rad_str).IP_displ[pSPARC->n_atom];
        for (int i = 0; i < ncol_total; i++){
            for (int j =0; j < (pSPARC->AO_rad_str).IP_displ[pSPARC->n_atom]; j++){
                fprintf(fp, "%.6E ", creal(gathered_alpha[count3]));
                count3++;
            }
            fprintf(fp, "\n");
        }

        fprintf(fp, "\n");
        count3=0;
        for (int i = 0; i < (pSPARC->AO_rad_str).IP_displ[pSPARC->n_atom] ; i++){
            for (int j =0; j < ncol_total; j++){
                fprintf(fp, "%.6E ", cimag(gathered_alpha[count3]));
                count3++;
            }
            fprintf(fp, "\n");
        }

        fclose(fp);
        free(gathered_alpha);
    }
    if (number==0){
        MPI_Comm_free(&new_comm);
    }
    
    free(alpha);

}


void get_DOS(SPARC_OBJ *pSPARC, double **DOS, int NDOS, int N_AO){
    int rank, rank_spincomm, rank_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);

     // only root processes of kptcomms will enter
    if (pSPARC->kptcomm_index < 0 || rank_kptcomm != 0) return; 

    int Nk = pSPARC->Nkpts_kptcomm;
    int Ns = pSPARC->Nstates;
    double occfac = 2.0/pSPARC->Nspin/pSPARC->Nspinor;
    // number of kpoints assigned to each kptcomm
    int    *Nk_i   = (int    *)malloc(pSPARC->npkpt * sizeof(int)); 
    double *kred_i = (double *)malloc(pSPARC->Nkpts_sym * 3 * sizeof(double));
    int *kpt_displs= (int    *)malloc((pSPARC->npkpt+1) * sizeof(int));

    char EigenFilename[L_STRING], AO_projFilename[L_STRING];
    if (rank == 0) snprintf(EigenFilename, L_STRING, "%s", "DOS.txt");
    if (rank == 0) snprintf(AO_projFilename, L_STRING, "%s", "Atomic_Projections.txt");

    FILE *output_fp;
    // first create an empty file
    if (rank == 0) {
        output_fp = fopen(EigenFilename,"w");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",EigenFilename);
            exit(EXIT_FAILURE);
        } 
        fprintf(output_fp, "Final Density of states\n");
        fclose(output_fp);   
    }

    // gather all eigenvalues and occupation number to root process in spin 
    int sendcount, *recvcounts, *displs;
    double *recvbuf_eig, *recvbuf_occ;
    sendcount = 0;
    recvcounts = NULL;
    displs = NULL;
    recvbuf_eig = NULL;
    recvbuf_occ = NULL;

    // first collect eigval/occ over spin
    if (pSPARC->npspin > 1) {
        // set up receive buffer and receive counts in kptcomm roots with spin up
        if (pSPARC->spincomm_index == 0) { 
            recvbuf_eig = (double *)malloc(pSPARC->Nspin * Nk * Ns * sizeof(double));
            recvbuf_occ = (double *)malloc(pSPARC->Nspin * Nk * Ns * sizeof(double));
            recvcounts  = (int *)   malloc(pSPARC->npspin * sizeof(int)); // npspin is 2
            displs      = (int *)   malloc((pSPARC->npspin+1) * sizeof(int)); 
            int i;
            displs[0] = 0;
            for (i = 0; i < pSPARC->npspin; i++) {
                recvcounts[i] = pSPARC->Nspin_spincomm * Nk * Ns;
                displs[i+1] = displs[i] + recvcounts[i];
            }
        } 
        // set up send info
        sendcount = pSPARC->Nspin_spincomm * Nk * Ns;
        MPI_Gatherv(pSPARC->lambda_sorted, sendcount, MPI_DOUBLE,
                    recvbuf_eig, recvcounts, displs,
                    MPI_DOUBLE, 0, pSPARC->spin_bridge_comm);
        MPI_Gatherv(pSPARC->occ_sorted, sendcount, MPI_DOUBLE,
                    recvbuf_occ, recvcounts, displs,
                    MPI_DOUBLE, 0, pSPARC->spin_bridge_comm);
        if (pSPARC->spincomm_index == 0) { 
            free(recvcounts);
            free(displs);
        }
    } else {
        recvbuf_eig = pSPARC->lambda_sorted;
        recvbuf_occ = pSPARC->occ_sorted;
    }

    double *eig_all = NULL, *occ_all = NULL;
    int *displs_all;
    displs_all = (int *)malloc((pSPARC->npkpt+1) * sizeof(int));  

    // next collect eigval/occ over all kpoints
    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {
        // set up receive buffer and receive counts in kptcomm roots with spin up
        if (pSPARC->kptcomm_index == 0) {
            int i;
            eig_all = (double *)malloc(pSPARC->Nspin * pSPARC->Nkpts_sym * Ns * sizeof(double));
            occ_all = (double *)malloc(pSPARC->Nspin * pSPARC->Nkpts_sym * Ns * sizeof(double));
            recvcounts = (int *)malloc(pSPARC->npkpt * sizeof(int));
            // collect all the number of kpoints assigned to each kptcomm
            MPI_Gather(&Nk, 1, MPI_INT, Nk_i, 1, MPI_INT,
               0, pSPARC->kpt_bridge_comm);
            displs_all[0] = 0;
            for (i = 0; i < pSPARC->npkpt; i++) {
                recvcounts[i] = Nk_i[i] * pSPARC->Nspin * Ns;
                displs_all[i+1] = displs_all[i] + recvcounts[i];
            }
            // collect all the kpoints assigend to each kptcomm
            // first set up sendbuf and recvcounts
            double *kpt_sendbuf = (double *)malloc(Nk * 3 * sizeof(double));
            int *kpt_recvcounts = (int *)malloc(pSPARC->npkpt * sizeof(int));
            // int *kpt_displs     = (int *)malloc((pSPARC->npkpt+1) * sizeof(int));
            if (pSPARC->BandStructFlag == 1) {
                for (i = 0; i < Nk; i++) {
                    kpt_sendbuf[3*i  ] = pSPARC->k1_inpt_kpt[i];
                    kpt_sendbuf[3*i+1] = pSPARC->k2_inpt_kpt[i];
                    kpt_sendbuf[3*i+2] = pSPARC->k3_inpt_kpt[i];
                }
            } else {
                for (i = 0; i < Nk; i++) {
                    kpt_sendbuf[3*i  ] = pSPARC->k1_loc[i]*pSPARC->range_x/(2.0*M_PI);
                    kpt_sendbuf[3*i+1] = pSPARC->k2_loc[i]*pSPARC->range_y/(2.0*M_PI);
                    kpt_sendbuf[3*i+2] = pSPARC->k3_loc[i]*pSPARC->range_z/(2.0*M_PI);
                }
            }
            kpt_displs[0] = 0; 
            for (i = 0; i < pSPARC->npkpt; i++) {
                kpt_recvcounts[i]  = Nk_i[i] * 3;
                kpt_displs[i+1] = kpt_displs[i] + kpt_recvcounts[i];
            }
            // collect reduced kpoints from all kptcomms
            MPI_Gatherv(kpt_sendbuf, Nk*3, MPI_DOUBLE, 
                kred_i, kpt_recvcounts, kpt_displs, 
                MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
            free(kpt_sendbuf);
            free(kpt_recvcounts);
        } else {
            // collect all the number of kpoints assigned to each kptcomm
            MPI_Gather(&Nk, 1, MPI_INT, Nk_i, 1, MPI_INT,
               0, pSPARC->kpt_bridge_comm);
            // collect all the kpoints assigend to each kptcomm
            double *kpt_sendbuf = (double *)malloc(Nk * 3 * sizeof(double));
            int kpt_recvcounts[1]={0}, i;
            if (pSPARC->BandStructFlag == 1) {
                for (i = 0; i < Nk; i++){
                    kpt_sendbuf[3*i  ] = pSPARC->k1_inpt_kpt[i];
                    kpt_sendbuf[3*i+1] = pSPARC->k2_inpt_kpt[i];
                    kpt_sendbuf[3*i+2] = pSPARC->k3_inpt_kpt[i];
                }
            } else {
                for (i = 0; i < Nk; i++) {
                    kpt_sendbuf[3*i  ] = pSPARC->k1_loc[i]*pSPARC->range_x/(2.0*M_PI);
                    kpt_sendbuf[3*i+1] = pSPARC->k2_loc[i]*pSPARC->range_y/(2.0*M_PI);
                    kpt_sendbuf[3*i+2] = pSPARC->k3_loc[i]*pSPARC->range_z/(2.0*M_PI);
                }
            }
            // collect reduced kpoints from all kptcomms
            MPI_Gatherv(kpt_sendbuf, Nk*3, MPI_DOUBLE, 
                kred_i, kpt_recvcounts, kpt_displs, 
                MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
            free(kpt_sendbuf);
        }   
        // set up send info
        sendcount = pSPARC->Nspin * Nk * Ns;
        MPI_Gatherv(recvbuf_eig, sendcount, MPI_DOUBLE,
                    eig_all, recvcounts, displs_all,
                    MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
        MPI_Gatherv(recvbuf_occ, sendcount, MPI_DOUBLE,
                    occ_all, recvcounts, displs_all,
                    MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
        if (pSPARC->kptcomm_index == 0) {
            free(recvcounts);
            //free(displs_all);
        }
    } else {
        int i;
        Nk_i[0] = Nk; // only one kptcomm
        kpt_displs[0] = 0;
        displs_all[0] = 0;
        if (pSPARC->BC != 1) {
            if(pSPARC->BandStructFlag == 1) {
                for (i = 0; i < Nk; i++) {
                    kred_i[3*i  ] = pSPARC->k1_inpt_kpt[i];
                    kred_i[3*i+1] = pSPARC->k2_inpt_kpt[i];
                    kred_i[3*i+2] = pSPARC->k3_inpt_kpt[i];
                }
            } else {
                for (i = 0; i < Nk; i++) {
                    kred_i[3*i  ] = pSPARC->k1_loc[i]*pSPARC->range_x/(2.0*M_PI);
                    kred_i[3*i+1] = pSPARC->k2_loc[i]*pSPARC->range_y/(2.0*M_PI);
                    kred_i[3*i+2] = pSPARC->k3_loc[i]*pSPARC->range_z/(2.0*M_PI);
                }
            }
        } else {
            kred_i[0] = kred_i[1] = kred_i[2] = 0.0;
        }
        eig_all = recvbuf_eig;
        occ_all = recvbuf_occ;
    }

    double min_E = 0.0, max_E = 0.0, delta_E = 0.1/27.211386245988, coeff = 1.0/(delta_E*sqrt(2*M_PI));

    if (pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            DOS = (double **) malloc (NDOS *sizeof(double*));
            for (int i = 0; i < NDOS; i++){
                DOS[i] = (double *) malloc (2 *sizeof(double));
            }
        }
    }


    // let root process print eigvals and occs to .eigen file
    if (pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            int k, Kcomm_indx, i;
            if (pSPARC->Nspin == 1) {
                for (Kcomm_indx = 0; Kcomm_indx < pSPARC->npkpt; Kcomm_indx++) {
                    int Nk_Kcomm_indx = Nk_i[Kcomm_indx];
                    for (k = 0; k < Nk_Kcomm_indx; k++) {
                        int kred_index = kpt_displs[Kcomm_indx]/3+k+1;
                        for (i = 0; i < pSPARC->Nstates; i++) {
                            min_E = min(min_E, eig_all[displs_all[Kcomm_indx] + k*Ns + i]);
                            max_E = max(max_E, eig_all[displs_all[Kcomm_indx] + k*Ns + i]);
                        }
                    }
                }
            } else if (pSPARC->Nspin == 2) {
                for (Kcomm_indx = 0; Kcomm_indx < pSPARC->npkpt; Kcomm_indx++) {
                    int Nk_Kcomm_indx = Nk_i[Kcomm_indx];
                    for (k = 0; k < Nk_Kcomm_indx; k++) {
                        int kred_index = kpt_displs[Kcomm_indx]/3+k+1;
                        for (i = 0; i < pSPARC->Nstates; i++) {
                            
                            min_E = min(min_E, eig_all[displs_all[Kcomm_indx] + k*Ns + i]);
                            max_E = max(max_E, eig_all[displs_all[Kcomm_indx] + k*Ns + i]);
                        }
                    }
                }
            }
        }
    }

    if (pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            double dE = (max_E-min_E)/(NDOS-1);
            for (int i = 0; i < NDOS; i++){
                DOS[i][0] = min_E + i*dE;
            }
        }
    }

    srand(time(NULL)); 
    // let root process print eigvals and occs to .eigen file
    if (pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            output_fp = fopen(AO_projFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",EigenFilename);
                exit(EXIT_FAILURE);
            }
            
            int k, Kcomm_indx, i;
            if (pSPARC->Nspin == 1) {
                for (Kcomm_indx = 0; Kcomm_indx < pSPARC->npkpt; Kcomm_indx++) {
                    int Nk_Kcomm_indx = Nk_i[Kcomm_indx];
                    for (k = 0; k < Nk_Kcomm_indx; k++) {
                        int kred_index = kpt_displs[Kcomm_indx]/3+k+1;
                        for (i = 0; i < pSPARC->Nstates; i++) {
                            double lambda_val = eig_all[displs_all[Kcomm_indx] + k*Ns + i];
                            for (int iAO = 0; iAO < N_AO; iAO++){
                                fprintf(output_fp, "%.9E ", (double)rand() / RAND_MAX);
                            }
                            fprintf(output_fp,"\n");

                            for (int iE = 0; iE < NDOS; iE++){
                                DOS[iE][1] += coeff*((pSPARC->kptWts[kred_index-1]+0.0)/pSPARC->Nkpts) * exp(-0.5*pow((lambda_val-DOS[iE][0])/delta_E, 2.0));
                            }
                        }
                    }
                }
            } else if (pSPARC->Nspin == 2) {
                for (Kcomm_indx = 0; Kcomm_indx < pSPARC->npkpt; Kcomm_indx++) {
                    int Nk_Kcomm_indx = Nk_i[Kcomm_indx];
                    for (k = 0; k < Nk_Kcomm_indx; k++) {
                        int kred_index = kpt_displs[Kcomm_indx]/3+k+1;
                        for (i = 0; i < pSPARC->Nstates; i++) {
                            double lambda_val = eig_all[displs_all[Kcomm_indx] + k*Ns + i];

                            for (int iAO = 0; iAO < N_AO; iAO++){
                                fprintf(output_fp, "%.9E ", (double)rand() / RAND_MAX);
                            }
                            fprintf(output_fp,"\n");

                            for (int iE = 0; iE < NDOS; iE++){
                                DOS[iE][1] += coeff*((pSPARC->kptWts[kred_index-1]+0.0)/pSPARC->Nkpts) * exp(-0.5*pow((lambda_val-DOS[iE][0])/delta_E, 2.0));
                            }
                        }
                    }
                }
            }
            fclose(output_fp);
        }
    }

    if (pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            output_fp = fopen(EigenFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",EigenFilename);
                exit(EXIT_FAILURE);
            }

            for (int i = 0; i < NDOS; i++){
                fprintf(output_fp, "%.9E    %.9E\n", DOS[i][0], DOS[i][1]);
            }
            fclose(output_fp);
        }
    }




    free(Nk_i);
    free(kred_i);
    free(kpt_displs);
    free(displs_all);

    if (pSPARC->npspin > 1) {
        if (pSPARC->spincomm_index == 0) { 
            free(recvbuf_eig);
            free(recvbuf_occ);
        }
    }

    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            free(eig_all);
            free(occ_all);
        }
    }
}

void get_PDOS(SPARC_OBJ *pSPARC, double **DOS, int NDOS, int N_AO){
    int rank, rank_spincomm, rank_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);

    FILE *output_fp;

     // only root processes of kptcomms will enter
    if (pSPARC->kptcomm_index < 0 || rank_kptcomm != 0) return;

    char EigenFilename[L_STRING];
    if (rank == 0) snprintf(EigenFilename, L_STRING, "%s", "PDOS.txt");

    if (pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            output_fp = fopen(EigenFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",EigenFilename);
                exit(EXIT_FAILURE);
            }

            for (int i = 0; i < NDOS; i++){
                fprintf(output_fp, "%.9E ", DOS[i][0]);
                for (int j = 0; j < N_AO; j++){
                    fprintf(output_fp, "%.9E ", DOS[i][1]/N_AO);
                }
                fprintf(output_fp,"\n");
            }
            fclose(output_fp);
        }
    }

}


void get_cohp(SPARC_OBJ *pSPARC, double **DOS, int NDOS, int N_AO){
    int rank, rank_spincomm, rank_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);

    FILE *output_fp;

     // only root processes of kptcomms will enter
    if (pSPARC->kptcomm_index < 0 || rank_kptcomm != 0) return;

    char EigenFilename[L_STRING];
    if (rank == 0) snprintf(EigenFilename, L_STRING, "%s", "pCOHP.txt");

    if (pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            output_fp = fopen(EigenFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",EigenFilename);
                exit(EXIT_FAILURE);
            }

            for (int i = 0; i < NDOS; i++){
                fprintf(output_fp, "%.9E ", DOS[i][0]);
                for (int j = 0; j < N_AO*(1+N_AO)/2; j++){
                    fprintf(output_fp, "%.9E ", DOS[i][1]/N_AO/N_AO);
                }
                fprintf(output_fp,"\n");
            }
            fclose(output_fp);
        }
    }
    
}

void print_DOS_COHP(SPARC_OBJ *pSPARC){
    double **DOS;
    int NDOS = 5000;
    read_upf_AO(pSPARC);
    int N_AO = 0;
    for (int i = 0; i < pSPARC->Ntypes; i++){
        N_AO += (pSPARC->AO_rad_str).num_orbitals[i] * pSPARC->nAtomv[i];
    }

    get_DOS(pSPARC, DOS, NDOS, N_AO);
    get_PDOS(pSPARC, DOS, NDOS, N_AO);
    get_cohp(pSPARC, DOS, NDOS, N_AO);

    if (pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            for (int i = 0; i < NDOS; i++){
                free(DOS[i]);
            }
            free(DOS);
        }
    }

}