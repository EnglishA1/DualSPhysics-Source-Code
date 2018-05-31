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

/// \file JSphCpu.h \brief Declares the class \ref JSphCpu.

#ifndef _JSphCpu_
#define _JSphCpu_

#include "Types.h"
#include "JSphTimersCpu.h"
#include "JSph.h"
#include <string>

class JCellDivCpu;
class JPartsOut;
class JPeriodicCpu;

//##############################################################################
//# JSphCpu
//##############################################################################
/// \brief Defines the attributes and functions to be used only in CPU simulations.

class JSphCpu : public JSph
{
private:
  JCellDivCpu* CellDiv;
  const JPeriodicCpu* PeriodicZone;

  //-Pointers to data of particles involved in the interaction: ComputeForces(),ComputeForcesShepard().
  //-The use of "i" and "j" indexes is because of the need to select particles from different blocks when using periodic boundary conditions.
  const unsigned *Idpi;      ///<Identifier of particle i according to its position in data involved in the interaction.
  const unsigned *Idpj;      ///<Identifier of particle j according to its position in data involved in the interaction.
  const word *Codei;         ///<Code with type and sub-type of the particle i involved in the interaction.
  const word *Codej;         ///<Code with type and sub-type of the particle j involved in the interaction.
  const tfloat3 *Posi;       ///<Position (X,Y,Z) of the particle i involved in the interaction.
  const tfloat3 *Posj;       ///<Position (X,Y,Z) of the particle j involved in the interaction.
  const tfloat3 *Veli;       ///<Velocity (X,Y,Z) of the particle i involved in the interaction.
  const tfloat3 *Velj;       ///<Velocity (X,Y,Z) of the particle j involved in the interaction.
  const float *Rhopi;        ///<Density of the particle i involved in the interaction.
  const float *Rhopj;        ///<Density of the particle j involved in the interaction.
  const float *PrRhopi;      ///<Pressure term of particle i (Pres[]/(Rhop[]*Rhop[]) involved in the interaction.
  const float *PrRhopj;      ///<Pressure term of particle j (Pres[]/(Rhop[]*Rhop[]) involved in the interaction.
  const float *Csoundi;      ///<Speed of sound of particle i Cs0*pow((Rhop[]*OVERRHOPZERO),3) involved in the interaction.
  const float *Csoundj;      ///<Speed of sound of particle j Cs0*pow((Rhop[]*OVERRHOPZERO),3) involved in the interaction.
  const float *Tensili;      ///<Term for tensile correction with \ref KERNEL_Cubic of the particle i involved in the interaction.
  const float *Tensilj;      ///<Term for tensile correction with \ref KERNEL_Cubic of the particle j involved in the interaction.
  const tsymatrix3f *Taui;   ///<SPS sub-particle stress tensor of the particle i involved in the interaction.
  const tsymatrix3f *Tauj;   ///<SPS sub-particle stress tensor of the particle j involved in the interaction.

  void InitVars();

protected:
  int OmpThreads;            ///<Maximum number of threads for OpenMP CPU executions.
  TpOmpMode OmpMode;         ///<Type of execution with or without OpenMP.

  std::string RunMode;       ///<Stores execution mode (symmetry,openmp,balancing,...).
  float Scell;               ///<Size of cell: 2h or h.
  float MovLimit;            ///<Maximum distance a particle can travel in one step.

  //-Amount of particles of the domain at each step.
  unsigned Np;               ///<Number of total particles at that step.
  unsigned Npb;              ///<Number of boundary particles at that step.
  unsigned NpbOk;            ///<Number of boundary particles that interact with fluid particles at that step. 
  
  bool WithFloating;         ///<Indicates if there are floating bodies.
  bool BoundChanged;         ///<Indicates if a boundary particle has moved since last step.

  JPartsOut* PartsOut;       ///<Number of excluded particles (out).

  unsigned ParticlesSize;    ///<Number of particles for which CPU memory was allocated.
  long long MemCpuParticles; ///<Allocated memory in CPU for the arrays with particle data.

  unsigned *RidpMoving;      ///<Position in data according to the particle identifier of boundary moving particles [CaseNmoving].
   
  //-Pointers to data of particles for the execution (size=ParticlesSize).
  unsigned *Idp;             ///<Identifier of the particles according to their position in data.
  word *Code;                ///<Code with type and sub-type of the particles. 
  tfloat3 *Pos;              ///<Position (X,Y,Z) of the particles.
  tfloat3 *Vel;              ///<Velocity (X,Y,Z) of the particles. 
  float *Rhop;               ///<Density of the particles.

  //-Variables in force computation.                            
  tfloat3 *Ace;              ///<Acceleration of the particles (X,Y,Z).         [\ref INTER_Forces,\ref INTER_ForcesCorr]
  float *Ar;                 ///<Density derivative.                            [\ref INTER_Forces,\ref INTER_ForcesCorr]
  tfloat3 *VelXcor;          ///<XSPH correction of velocity (without Eps).     [\ref INTER_Forces]
  float *FdWab;              ///<Kernel summation for Shepard Filter.           [\ref INTER_Shepard]
  float *FdRhop;             ///<Density summation for Shepard Filter.          [\ref INTER_Shepard]
  float *Delta;              ///<Approach Delta-SPH [\ref INTER_Forces]
    
  //-Variables used in the force computation.
  float *Csound;             ///<Speed of sound: Cs0*pow((Rhop[]*OVERRHOPZERO),3).
  float *PrRhop;             ///<Pressure term: Pres[]/(Rhop[]*Rhop[]).
  float *Tensil;             ///<Term for tensile correction: only with \ref KERNEL_Cubic.  
  
  //-Variables used to update system using VERLET algorithm.
  tfloat3 *VelM1;            ///<Verlet: array to store velocity values of the previous time step.
  float *RhopM1;             ///<Verlet: array to store density values of the previous time step.
  int VerletStep;            ///<Current step of the Verlet algorithm after having applied Eulerian equations.

  //-Variables used to update system using SYMPLECTIC algorithm.
  tfloat3 *PosPre;           ///<Sympletic: array to store position values in predictor step.
  tfloat3 *VelPre;           ///<Sympletic: array to store velocity values in predictor step.
  float *RhopPre;            ///<Sympletic: array to store density values in predictor step.
  float DtPre;               ///<Sympletic: array to store time step value in predictor step.

  //-Variables for Laminar+SPS viscosity.  
  tsymatrix3f *Tau;          ///<SPS sub-particle stress tensor.
  tsymatrix3f *Csph;         ///<Velocity gradients.

  //-Variables for floating bodies.
  unsigned *FtRidp;          ///<Identifier to access to the particles of the floating object [CaseNfloat].
  tfloat3* FtDist;           ///<Distance of the particles to the centre of the floating object [CaseNfloat].
   
  float CsoundMax;           ///<Maximum value of Csound[] computed in PreInteraction_Forces().
  
  //-viscdt is the value to compute new time step according to viscosity terms dt2=H/(cs_max+H*viscdt).
  float ViscDtMax;           ///<Maximum value of viscdt computed in Interaction_Forces().
  float ViscDtThread[MAXTHREADS_OMP*STRIDE_OMP]; ///<Variable to record the ViscDt of each thread (even without OpenMP).

  TimersCpu Timers;          ///<Declares an array with timers for CPU (type structure \ref StSphTimerCpu).

  TpParticle GetTpParticle(unsigned p)const{ return(GetTpParticleCode(Code[p])); }

  void FreeMemoryParticles();
  void AllocMemoryParticles(unsigned np);
  void FreeMemoryAuxParticles();
  void AllocMemoryAuxParticles(unsigned np,unsigned npf);
  long long GetAllocMemoryCpu()const;
  void PrintAllocMemory(long long mcpu)const;
  void ConfigOmp(const JCfgRun *cfg);
  void ConfigRunMode(const JCfgRun *cfg,const std::string &preinfo="");
  void UpdateMaxValues();
  void InitRun();

  void CalcRidp(unsigned n,unsigned ini,unsigned idini,unsigned idfin,const unsigned *idp,unsigned *ridp)const;
  void CalcRidpMoving(){ CalcRidp(Npb,0,CaseNfixed,CaseNfixed+CaseNmoving,Idp,RidpMoving); }

  void CalcFtRidp(){ CalcRidp(Np-Npb,Npb,CaseNpb,CaseNpb+CaseNfloat,Idp,FtRidp); }
  void InitFloating();

  void ConfigCellDiv(JCellDivCpu* celldiv){ CellDiv=celldiv; }
  
  void AddVarAcc();
  void PreInteraction_Forces(TpInter tinter);
  template<TpKernel tker> void PreInteraction_Forces_(TpInter tinter);
  void PreInteraction_Shepard();

  //-Methods for interaction between particles of the domain.
  void InteractionCells(TpInter tinter);
  template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void InteractionCells_Single();
  template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void InteractionCells_Static();
  template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void InteractionCells_Dynamic();
  template<bool tsym,TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void InteractSelf(int box,byte kind,byte kind2ini);
  template<bool tsym,TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void InteractCelij(int ibegin,int iend,int box1,int box2,byte kind2ini);

  //-Methods for interaction between particles of the domain with particles of PeriodicZone.
  void InteractionPeri(TpInter tinter,JPeriodicCpu* pzone);
  template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void InteractionPeri_Single();
  template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void InteractionPeri_Static();
  template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void InteractionPeri_Dynamic();
  template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> float InteractCelijPeri(unsigned i,TpParticle tpi,unsigned boxini,unsigned boxfin,byte kind2ini);

  void ConfigComputeDataij(const unsigned *idp,const word *code,const tfloat3 *pos,const float *rhop,const tfloat3 *vel,const float *prrhop,const float* csound,const float* tensil,const tsymatrix3f* tau){ 
    Idpi=Idp; Codei=Code; Posi=Pos; Rhopi=Rhop; Veli=Vel; PrRhopi=PrRhop; Csoundi=Csound; Tensili=Tensil; Taui=Tau;
    Idpj=idp; Codej=code; Posj=pos; Rhopj=rhop; Velj=vel; PrRhopj=prrhop; Csoundj=csound; Tensilj=tensil; Tauj=tau;
  }
  template<bool tsym,TpKernel tker> void ComputeForcesShepard(int i,int j,float drx,float dry,float drz,float rr2,int offsetsh);
  template<bool tsym,TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> float ComputeForces(TpParticle tpi,TpParticle tpj,int i,int j,float drx,float dry,float drz,float rr2,int offset,int offsetf);

  void ComputeVerletVars(const tfloat3 *vel1,const tfloat3 *vel2,float dt,float dt2,tfloat3 *velnew);
  void ComputeVerlet(bool rhopbound,float dt);
  void ComputeSymplecticPre(bool rhopbound,float dt);
  void ComputeSymplecticCorr(bool rhopbound,float dt);
  void ComputeRhop(float* rhopnew,const float* rhopold,float armul,bool rhopbound);
  void ComputeRhopEpsilon(float* rhopnew,const float* rhopold,float armul,bool rhopbound);
  void ComputeShepard(int pini,int pfin);

  float DtVariable();
  void RunMotion(float stepdt);

  void SPSCalcTau(); 

  void ShowTimers(bool onlyfile=false);
  void GetTimersInfo(std::string &hinfo,std::string &dinfo)const;
  unsigned TimerGetCount()const{ return(TmcGetCount()); }
  bool TimerIsActive(unsigned ct)const{ return(TmcIsActive(Timers,(CsTypeTimerCPU)ct)); }
  float TimerGetValue(unsigned ct)const{ return(TmcGetValue(Timers,(CsTypeTimerCPU)ct)); }
  const double* TimerGetPtrValue(unsigned ct)const{ return(TmcGetPtrValue(Timers,(CsTypeTimerCPU)ct)); }
  std::string TimerGetName(unsigned ct)const{ return(TmcGetName((CsTypeTimerCPU)ct)); }
  std::string TimerToText(unsigned ct)const{ return(JSph::TimerToText(TimerGetName(ct),TimerGetValue(ct))); }

  static void OmpMergeDataSum(int ini,int fin,float *data,int stride,int nthreads);
  static void OmpMergeDataSum(int ini,int fin,tfloat3 *data,int stride,int nthreads);
  static void OmpMergeDataSumError(int ini,int fin,float *data,int stride,int nthreads,float verror);

public:
  JSphCpu();
  ~JSphCpu();
};

#endif


