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

/// \file JSphGpu.h \brief Declares the class \ref JSphGpu.

#ifndef _JSphGpu_
#define _JSphGpu_

#include "Types.h"
#include "JSphTimersGpu.h"
#include "JSph.h"
#include <string>

class JPartsOut;
class JGpuArrays;
class JCellDivGpu;

//##############################################################################
//# JSphGpu
//##############################################################################
/// \brief Defines the attributes and functions used only in GPU simulations.

class JSphGpu : public JSph
{
private:
  JCellDivGpu* CellDiv;

public:
///Structure that stores the block size to be used in each interaction kernel during GPU execution.
  typedef struct {
    unsigned forcesfluid;
    unsigned forcesfluidcorr;
    unsigned forcesbound;
    unsigned shepard;
    unsigned forcesfluid_zoleft;
    unsigned forcesfluidcorr_zoleft;
    unsigned forcesbound_zoleft;
    unsigned shepard_zoleft;
    unsigned forcesfluid_zoright;
    unsigned forcesfluidcorr_zoright;
    unsigned forcesbound_zoright;
    unsigned shepard_zoright;
    unsigned forcesfluid_peri;
    unsigned forcesfluidcorr_peri;
    unsigned forcesbound_peri;
    unsigned shepard_peri;
  }StBlockSizes;

protected:
  std::string PtxasFile;     ///<File with information of number of registers to optimise the execution.
  StBlockSizes BlockSizes;   ///<Stores configuration of BlockSizes.
  std::string BlockSizesStr; ///<Stores configuration of BlockSizes in text format.

  std::string RunMode;       ///<Stores execution mode (symmetry,openmp,balancing,...).
  float Scell;               ///<Size of cell: 2h or h.
  float MovLimit;            ///<Maximum distance a particle can travel in one step.

  //-Amount of particle of the domain at each step.
  unsigned Np;               ///<Number of total particles at that step.
  unsigned Npb;              ///<Number of boundary particles at that step.
  unsigned NpbOk;            ///<Number of boundary particles that interact with fluid particles at that step. 
  
  bool WithFloating;         ///<Indicates if there are floating bodies.
  bool BoundChanged;         ///<Indicates if a boundary particle has moved since last step.
  bool Multiphase;			 // if multi-phase

  JPartsOut* PartsOut;       ///<Number of excluded particles (out).
  
  
  
  unsigned CpuParticlesSize; ///<Number of particles for which CPU memory was allocated.
  long long MemCpuParticles; ///<Allocated memory in CPU for the arrays with particle data.

  //-Pointers to data of particles in CPU for the execution (size=CpuParticlesSize).
  unsigned *Idp;             ///<Identifier of the particles according to their position in CPU data.
  word *Code;                ///<Code with type and sub-type of the particles in CPU. 
  tfloat3 *Pos;              ///<Position (X,Y,Z) of the particles in CPU.
  tfloat3 *Vel;              ///<Velocity (X,Y,Z) of the particles in CPU. 
  float *Rhop;               ///<Density of the particles in CPU.
  //tfloat4 *Velrhop;

  //Additional
  float *Press;				///Pressure
  float *Viscop;			//viscosity for each particle
  tsymatrix3f *Vtau;			// Viscous forces (dont really need this for outpout only for debug)
  //No CPU arrays for Vtau etc
  //multiphase arrays
  unsigned *Idpm;			///Phase store 0/1
  
  unsigned GpuParticlesSize; ///<Number of particles for which GPU memory was allocated.
  long long MemGpuParticles; ///<Allocated memory in GPU for the arrays with particle data.

  unsigned *RidpMoving;      ///<Position in data according to the particle identifier of boundary moving particles [CaseNmoving].

  JGpuArrays* GpuArrays;     ///<List of arrays in GPU of 1, 4 and 12 bytes for particles.

  //-Pointers to data of particles in GPU for the execution (size=GpuParticlesSize).
  unsigned *Idpg;            ///<Identifier of the particles according to their position in GPU data.
  word *Codeg;               ///<Code with type and sub-type of the particles in GPU. 
  float3 *Posg;              ///<Position (X,Y,Z) of the particles in GPU.
  float3 *Velg;              ///<Velocity (X,Y,Z) of the particles in GPU. 
  float *Rhopg;              ///<Density of the particles in GPU.
  
  //Additional
  float *Pressg;			///Pressure
  float *Viscopg;			//viscosity for each particle  
  float *CVg;				//Concentration Volumetric for the Vand eq
  float *Surfg;				//Shifting - surface tracking
  float3 *Cderivg;			//Shifting concentration grad
  //Viscous formulation
  tsymatrix3f *Vtaug;

  //multiphase arrays
  unsigned *Idpmg;			///Phase store 0/1

  unsigned Npl, Nps;
  unsigned Idbl,Idbs,Idbb,Idbf;
	      
  //-Variables in force computation in GPU.  
  float4 *PosPresg;          ///<Combination of position and pressure of the particle in GPU.
  float4 *VelRhopg;          ///<Combination of velocity and density of the particle in GPU.
  float *ViscDtg;            ///<Value to compute new time step according to viscosity terms dt2=H/(cs_max+H*viscdt) in GPU.
  float3 *Aceg;              ///<Accumulates acceleration of the particle (X,Y,Z) in GPU.
  float *Arg;                ///<Accumulates density derivative of the particle in GPU.
  float3 *VelXcorg;          ///<XSPH correction of velocity (without Eps) in GPU.
  float *FdWabg;             ///<Kernel summation for Shepard Filter in GPU.   
  float *FdRhopg;            ///<Density summation for Shepard Filter in GPU. 
  float4 *PosVolg;           ///<Combination of position and volume in Shepard Filter for GPU.
  float *Deltag;             ///<Accumulates Delta-SPH with DELTA_DBCExt in GPU.

  //-Variables used to update system using VERLET algorithm in GPU.
  float3 *VelM1g;            ///<Verlet: array to store velocity values of the previous time step in GPU.
  float *RhopM1g;            ///<Verlet: array to store density values of the previous time step in GPU.
  int VerletStep;            ///<Current step of the Verlet algorithm after having applied Eulerian equations in GPU.

  //-Variables used to update system using SYMPLECTIC algorithm in GPU.
  float3 *PosPreg;           ///<Sympletic: array to store position values in predictor step in GPU.
  float3 *VelPreg;           ///<Sympletic: array to store velocity values in predictor step in GPU.
  float *RhopPreg;           ///<Sympletic: array to store density values in predictor step in GPU.
  float DtPre;               ///<Sympletic: array to store time step value in predictor step in GPU.   

  //-Variables for Laminar+SPS viscosity in GPU.  
  tsymatrix3f *Taug;          ///<SPS sub-particle stress tensor in GPU.
  tsymatrix3f *Csphg;         ///<Velocity gradients in GPU.

  //-Variables for floating bodies in GPU.
  unsigned *FtRidpg;
  float3 *FtDistg;           ///<Identifier to access to the particles of the floating object [CaseNfloat] in GPU.
  float3 *FtOmegaVelg;       ///<Distance of the particles to the centre of the floating object [CaseNfloat] in GPU.
  float *FtoMasspg;          ///<Mass of the particle for each floating body [FtCount] in GPU.

  float CsoundMax;           ///<Maximum value of Csound[] computed in PreInteraction_Forces() in GPU.
  
  float ViscDtMax;           ///<Maximum value of viscdt computed in Interaction_Forces() in GPU.
  float ForceDtMax;
  float ViscoDtMax;
  float CsDtMax;

  TimersGpu Timers;          ///<Declares an array with timers for CPU (type structure \ref StSphTimerGpu).

  //-Variables with information of the GPU hardware.
  int GpuSelect;             ///<Selected GPU  (-1: no selection).
  std::string GpuName;       ///<Name of the selected GPU.
  size_t GpuGlobalMem;       ///<Global memory size in bytes.
  unsigned GpuSharedMem;     ///<Shared memory size per block in bytes.
  unsigned GpuCompute;       ///<Compute capability: 10,11,12,20.

  void InitVars();
  void RunExceptionCuda(const std::string &method,const std::string &msg,cudaError_t error);
  void CheckCudaError(const std::string &method,const std::string &msg);

  void FreeCpuMemoryParticles();
  void AllocCpuMemoryParticles(unsigned np);
  void FreeGpuMemoryParticles();
  void AllocGpuMemoryParticles(unsigned np,float over);
  void ReserveBasicGpuArrays();

  long long GetAllocMemoryCpu()const;
  long long GetAllocMemoryGpu()const;
  void PrintAllocMemory(long long mcpu,long long mgpu)const;

  void ParticlesDataUp(unsigned n);
  void ParticlesDataDown(unsigned n,unsigned pini,bool code,bool orderdecode);  //unsigned pini=0,bool checkout=false,bool orderdecode=false
  
  void SelecDevice(int gpuid);
  static unsigned OptimizeBlockSize(unsigned compute,unsigned nreg);
  unsigned BlockSizeConfig(const std::string& opname,unsigned compute,unsigned regs);
  void ConfigBlockSizes(bool usezone,bool useperi);
  void ConfigRunMode(const std::string &preinfo);
  void InitFloating();
  void InitRun();

  void ConfigCellDiv(JCellDivGpu* celldiv){ CellDiv=celldiv; }

  void AddVarAcc();
  void PreInteractionVars_Forces(TpInter tinter,unsigned ini,unsigned np,unsigned npb);
  void PreInteraction_Forces(TpInter tinter);
  void PosInteraction_Forces();
  
  void PreInteractionVars_Shepard(unsigned pini,unsigned pfin);
  void PreInteraction_Shepard(bool onestep);
  void PosInteraction_Shepard();

  void ComputeVerlet(bool rhopbound,float dt);
  void ComputeSymplecticPre(bool rhopbound,float dt);
  void ComputeSymplecticCorr(bool rhopbound,float dt);

  float DtVariable();
  void RunMotion(float stepdt);

  void ShowTimers(bool onlyfile=false);
  void GetTimersInfo(std::string &hinfo,std::string &dinfo)const;
  unsigned TimerGetCount()const{ return(TmgGetCount()); }
  bool TimerIsActive(unsigned ct)const{ return(TmgIsActive(Timers,(CsTypeTimerGPU)ct)); }
  float TimerGetValue(unsigned ct)const{ return(TmgGetValue(Timers,(CsTypeTimerGPU)ct)); }
  const double* TimerGetPtrValue(unsigned ct)const{ return(TmgGetPtrValue(Timers,(CsTypeTimerGPU)ct)); }
  std::string TimerGetName(unsigned ct)const{ return(TmgGetName((CsTypeTimerGPU)ct)); }
  std::string TimerToText(unsigned ct)const{ return(JSph::TimerToText(TimerGetName(ct),TimerGetValue(ct))); }

public:
  JSphGpu();
  ~JSphGpu();
  
  //------Debug-----
public:
  //void DgSaveVtkParticlesGpu(std::string filename,int numfile,unsigned pini,unsigned pfin,const float3 *posg,const byte *checkg=NULL,const unsigned *idpg=NULL,const float3 *velg=NULL,const float *rhopg=NULL);
  //void DgSaveCsvParticlesGpu(std::string filename,int numfile,unsigned pini,unsigned pfin,std::string head,const float3 *posg=NULL,const unsigned *idpg=NULL,const float3 *velg=NULL,const float *rhopg=NULL,const float *arg=NULL,const float3 *aceg=NULL,const float3 *vcorrg=NULL);

};

#endif


