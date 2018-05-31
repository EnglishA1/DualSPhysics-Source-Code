/*
<DUALSPHYSICS>  Copyright (C) 2013 by Jose M. Dominguez, Dr Alejandro Crespo, Prof. M. Gomez Gesteira, Anxo Barreiro, Ricardo Canelas
                                      Dr Benedict Rogers, Dr Stephen Longshaw, Dr Renato Vacondio

EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

This file is part of DualSPHysics. 

DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/// \file JSphGpu_ker.h \brief Declares functions and CUDA kernels for the particle interaction and system update.

#ifndef _JSphGpu_ker_
#define _JSphGpu_ker_

#include "Types.h"
#include "JSphTimersGpu.h"
#include <cuda_runtime_api.h>

//#define DG_ReduMaxFloat //-In ReduMaxFloat() checks that the result is correct.

class JLog2;

#define SPHBSIZE 256

///Structure with constants stored in the constant memory of GPU for the particle interactions.
typedef struct{
  float massb;               ///<Mass of a boundary particle.
  float massf;               ///<Mass of a fluid particle.
  unsigned nbound;           ///<Number of boundary particles.
  float h;                   ///<Smoothing length (=coef*sqrt(dx*dx+dy*dy+dz*dz))
  float fourh2;              ///< \ref h * \ref h * 4 
  float cubic_a1,cubic_a2,cubic_aa,cubic_a24,cubic_c1,cubic_d1,cubic_c2,cubic_odwdeltap; ///<Ctes of Cubic Spline kernel.
  float wendland_awen,wendland_bwen;                                                     ///<Ctes of Wendland kernel.
  float cteshepard;          ///<Constant used in Shepard filter to evaluate the self contribution of the particle itself.
  float cs0;                 ///<Speed of sound of reference.
  float visco;               ///<Viscosity value.
  float eta2;                ///<eta*eta being eta=0.1*\ref h
  float overrhop0;
  float delta2h;             ///<delta2h=DeltaSph*H*2
  float posminx,posminy,posminz;
  float scell,dosh;
  //Additional
  unsigned phasecount;
  float spssmag;
  float spsblin;
  float minvisc;
}StCteInteraction;

/// Structure that holds multiphase constants
  typedef struct {
    byte mkfluid;
    float visco;
	float visco_cr;
	float visco_max;
	float coh;
	float phi;
	float hbp_m;
	float hb_n;
    unsigned idbegin;
    unsigned count;
	float DP_alpha;
	float DP_kappa;
	float ys;
	unsigned yc;
	unsigned sed;
  }StPhaseCteg;

  /// Structure that holds multiphase constants
  typedef struct {
    int phaseid;
	float rho_ph;
	float mass_ph;
	float Cs0_ph;
	float b_ph;
	float Gamma_ph;
  }StPhaseArrayg;


 /* StPhaseCteg* phasecte;
  StPhaseArrayg* phasearray;*/

/// Implements a set of functions and CUDA kernels for the particle interaction and system update.
namespace cusph{

inline float3 Float3(const tfloat3& v){ float3 p={v.x,v.y,v.z}; return(p); }
inline float3 Float3(float x,float y,float z){ float3 p={x,y,z}; return(p); }
inline tfloat3 ToTFloat3(const float3& v){ return(TFloat3(v.x,v.y,v.z)); }

dim3 GetGridSize(unsigned n,unsigned blocksize);
inline unsigned ReduMaxFloatSize(unsigned ndata){ return((ndata/SPHBSIZE+1)+(ndata/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)); }
float ReduMaxFloat(unsigned ndata,unsigned inidata,float* data,float* resu);

void CalcFa2(unsigned n,float3 *ace,float *fa2);
void CalcMiou(unsigned n,float *viscop,float *rhop,float *resu);

void CteInteractionUp(const StCteInteraction *cte, const StPhaseCteg *phasecte, const StPhaseArrayg *phasearray,unsigned phases);

void PreInteraction_Forces(unsigned np,unsigned npb,const float3 *pos,const float3 *vel,const float *rhop,float4 *pospres,float4 *velrhop,float3 *ace,tfloat3 gravity,float *press, const unsigned *idpm);
void PreYieldStress(unsigned np,unsigned npb,const float *rhop,float4 *pospres,const unsigned *idpmg,float *viscop);
void YieldResetAce(unsigned np,unsigned npb,tsymatrix3f *vtau,float *viscop,float3 *ace,unsigned *idpm, float3 *vel,float *ar);
//void YieldResetAce(unsigned np,unsigned npb,tsymatrix3f *vtau,float *viscop,float3 *ace,unsigned *idpm, float4 *velrhop);

void SPSCalcTau(bool floating,unsigned np,unsigned npb,float smag,float blin,const float *rhop,const word *code,const tsymatrix3f *csph,tsymatrix3f *tau);


//multi
void MultiInteraction_Forces(TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,TpCellMode cellmode,unsigned bsbound,unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells,const unsigned *cellpart,const int2 *begincell,const float4 *pospres,const float4 *velrhop,const unsigned *idp, tsymatrix3f* tau,tsymatrix3f* vtau,float *viscdt,float* ar,float3 *ace,float3 *velxcor,tsymatrix3f* csph,float *delta,const unsigned *idpm,float *viscop,float *cv,bool simulate2d);

void Interaction_Forces(TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,TpCellMode cellmode,unsigned bsbound,unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells,const unsigned *cellpart,const int2 *begincell,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f* tau,float *viscdt,float* ar,float3 *ace,float3 *velxcor,tsymatrix3f* csph,float *delta,bool simulate2d);

void InteractionPeri_Forces(TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,TpCellMode cellmode,unsigned bsbound,unsigned bsfluid,unsigned np,unsigned npb,const unsigned *list,unsigned listbini,unsigned listfini,const unsigned *cellpart,unsigned ncx,unsigned cellfluid,const int2 *zobegincell,const float4 *zopospres,const float4 *zovelrhop,const unsigned *zoidp,const tsymatrix3f *zotau,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f *tau,float *viscdt,float *ar,float3 *ace,float3 *velxcor,tsymatrix3f *csph,float *delta,bool simulate2d);

void AddForceFluid(unsigned n,unsigned pini,word tpvalue,tfloat3 force,const word *code,float3 *ace);

void AddDelta(unsigned n,const float *delta,float *ar);

void PreInteraction_Shepard(bool floating,unsigned pini,unsigned pfin,const float3 *pos,const float *rhop,const unsigned *idp,unsigned nbound,float ftposout,float massf,float4 *posvol);
void Interaction_Shepard(TpKernel tkernel,TpCellMode cellmode,unsigned bsshepard,unsigned pini,unsigned pfin,tuint3 ncells,const unsigned *cellpart,const int2 *begincell,const float4 *posvol,float *fdrhop,float *fdwab);
void InteractionPeri_Shepard(TpKernel tkernel,TpCellMode cellmode,unsigned bsshepard,unsigned npf,const unsigned *list,unsigned listpini,const unsigned *cellpart,unsigned ncx,unsigned cellfluid,const int2 *zobegincell,const float4 *zoposvol,const float4 *posvol,float *fdrhop,float *fdwab);

void Compute_Shepard(unsigned pini,unsigned pfin,float massf,const float *fdrhop,const float *fdwab,float *rhop);

void ComputeStepVerlet(bool rhopbound,bool floating,unsigned np,unsigned npb,const float3 *vel1,const float3 *vel2,const float *rhop,const unsigned *idp,const float *ar,const float3 *ace,const float3 *velxcor,float dt,float dt2,float eps,float movlimit,float3 *pos,word *code,float3 *velnew,float *rhopnew,float rhop0);
void ComputeStepSymplecticPre(bool rhopbound,bool floating,unsigned np,unsigned npb,const unsigned *idp,const float3 *pospre,const float3 *velpre,const float *rhoppre,const float *ar,const float3 *velxcor,const float3 *ace,float dtm,float eps,float movlimit,float3 *pos,word *code,float3 *vel,float *rhop,float rhop0);
void ComputeStepSymplecticCor(bool rhopbound,bool floating,unsigned np,unsigned npb,const unsigned *idp,const float3 *pospre,const float3 *velpre,const float *rhoppre,const float *ar,const float3 *ace,float dtm,float dt,float movlimit,float3 *pos,word *code,float3 *vel,float *rhop,float rhop0);

void CalcRidp(unsigned np,unsigned pini,unsigned idini,unsigned idfin,const unsigned *idp,unsigned *ridp);

void MoveLinBound(bool simulate2d,unsigned np,unsigned ini,tfloat3 mvpos,tfloat3 mvvel,const unsigned *ridp,float3 *pos,float3 *vel,word *code,float movlimit);
void MoveMatBound(bool simulate2d,unsigned np,unsigned ini,tmatrix4f m,float dt,const unsigned *ridpmv,float3 *pos,float3 *vel,word *code,float movlimit);

void FtCalcDist(unsigned n,unsigned pini,tfloat3 center,const unsigned *ridpft,const float3 *pos,float3 *ftdist);
void FtCalcOmega(unsigned n,unsigned pini,tfloat3 gravity,float ftmass,const unsigned *ftridp,const float3 *ftdist,const float3 *ace,float3 *result);
void FtUpdate(bool predictor,unsigned n,unsigned pini,float dt,tfloat3 center,tfloat3 fvel,tfloat3 fomega,const unsigned *ftridp,float3 *ftdist,float3 *pos,float3 *vel);
void FtInteraction_Forces(TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,TpCellMode cellmode,unsigned bsbound,unsigned bsfluid,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells,const unsigned *cellpart,const int2 *begincell,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f* tau,const word *code,const float *ftomassp,float *viscdt,float* ar,float3 *ace,float3 *velxcor,tsymatrix3f* csph,float *delta,bool simulate2d);
unsigned FtFindFirstFloating(unsigned np,unsigned npb,const unsigned *list,unsigned listfini,const word *code,const unsigned *idp);

void AddVarAcc(unsigned n,unsigned pini,word codesel,tfloat3 acclin,tfloat3 accang,tfloat3 centre,const word *code,const float3 *pos,float3 *ace);

}

#endif



