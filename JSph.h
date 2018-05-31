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

/// \file JSph.h \brief Declares the class \ref JSph.

#ifndef _JSph_
#define _JSph_

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

#include "Types.h"
#include "JObject.h"
#include "JCfgRun.h"
#include "JLog2.h"
#include "JTimer.h"
#include <float.h>
#include <string>
#include <cmath>
#include <ctime>
#include <sstream>
#include <iostream>
#include <fstream>

#include "JLog2.h"

#ifdef _WITHOMP
  #include <omp.h>
#endif

class JSphMotion;
class JPartData;
class JPartPData;
class JFloatingData;
class JSphDtFixed;
class JSpaceParts;
class JSphVarAcc;

class JPartDataBi4;
class JPartOutBi4Save;
class JPartFloatBi4Save;
class JPartsOut;
class JXml;

//##############################################################################
//# JSph
//##############################################################################
/// \brief Defines all the attributes and functions that CPU and GPU simulations share.

class JSph : protected JObject
{
public:
  static const unsigned int VersionMajor=300;
  static const unsigned int VersionMinor=0;
  static std::string GetVersionStr();

/// Structure with constants for the Cubic Spline kernel.
  typedef struct {
    float a1,a2,aa,a24,c1,d1,c2;
    float od_wdeltap;        ///<Parameter for tensile instability correction.  
  }StCubicCte;

/// Structure with constants for the Wendland kernel.
  typedef struct {
    float awen,bwen;
  }StWendlandCte;

/// Structure with MK information.
  typedef struct {
    unsigned begin;
    unsigned count;
    unsigned mktype;
    unsigned mk;
    word code;
  }StMkInfo;

  /// Structure that saves extra information about the execution.
  typedef struct {
    double timesim;     //-Segundos desde el inicio de la simulacion (despues de cargar los datos iniciales).       ///<Seconds from the start of the simulation (after loading the initial data).
    unsigned nct;       //-Numero de celdas usadas en el divide.                                                    ///<Number of cells used in the divide.
    unsigned npbin;     //-Numero de particulas bound dentro del area del divide (incluye particulas periodicas).   ///<Number of boundary particles within the area of the divide (includes periodic particles).
    unsigned npbout;    //-Numero de particulas bound fuera del area del divide (incluye particulas periodicas).    ///<Number of boundary particles outside of the area of the divide (includes periodic particles).
    unsigned npf;       //-Numero de particulas fluid (incluye particulas periodicas).                              ///<Number of fluid particles (includes periodic particles).
    unsigned npbper;    //-Numero de particulas bound periodicas (dentro y fuera del area del divide).              ///<Number of periodic boundary particles (inside and outside the area of the split).
    unsigned npfper;    //-Numero de particulas fluid periodicas.                                                   ///<Number of periodic fluid particles.
    unsigned newnpok;   //-Numero de nuevas particulas fluid (inlet conditions).                                    ///<Number of new fluid particles (inlet conditions)
    unsigned newnpfail; //-Numero de nuevas particulas fluid descartadas (inlet conditions).                        ///<Number of discarded new fluid particles (inlet conditions)
    llong memorycpualloc;
    bool gpudata;
    llong memorynpalloc;
    llong memorynpused;
    llong memorynctalloc;
    llong memorynctused;
  }StInfoPartPlus;

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
  }StPhaseCte;

  /// Structure that holds multiphase constants
  typedef struct {
    int phaseid;
	float rho_ph;
	float mass_ph;
	float Cs0_ph;
	float b_ph;
	float Gamma_ph;
  }StPhaseArray;


private:
  //-Variables of configuration to compute limits of the case.
  bool CfgDomainParticles;
  tfloat3 CfgDomainParticlesMin,CfgDomainParticlesMax;
  tfloat3 CfgDomainParticlesPrcMin,CfgDomainParticlesPrcMax;
  tfloat3 CfgDomainFixedMin,CfgDomainFixedMax;

  //-Objects to store info of particles in files.
  JPartData* BinData;  //-Format of bi2.
  //v4
  JPartDataBi4 *DataBi4;           //-Para grabar particulas e info en formato bi4.         ///<To store particles and info in bi4 format.
  JPartOutBi4Save *DataOutBi4;     //-Para grabar particulas excluidas en formato bi4.      ///<To store excluded particles in bi4 format.
  JPartFloatBi4Save *DataFloatBi4; //-Para grabar datos de floatings en formato bi4.        ///<To store floating data in bi4 format.
  JPartsOut *PartsOut;         //-Almacena las particulas excluidas hasta su grabacion.     ///<Stores excluded particles until they are saved.

  //-Amount of excluded particles according to reason.
  unsigned OutPosCount;      ///<Amount of excluded particles due to its position out of the limits of the domain.
  unsigned OutRhopCount;     ///<Amount of excluded particles due to its value of density out of the desirable limits.
  unsigned OutMoveCount;     ///<Amount of excluded particles since the particle moves faster than the allowed vecocity.
  unsigned OutInitialCount;  ///<Number of excluded particles before simulation (included in OutPosCount).

  void InitVars();
  std::string CalcRunCode()const;
  void AddOutCount(unsigned outpos,unsigned outrhop,unsigned outmove){ OutPosCount+=outpos; OutRhopCount+=outrhop; OutMoveCount+=outmove; }
  void ClearCfgDomain();
  void ConfigDomainFixed(tfloat3 vmin,tfloat3 vmax);
  void ConfigDomainParticles(tfloat3 vmin,tfloat3 vmax);
  void ConfigDomainParticlesPrc(tfloat3 vmin,tfloat3 vmax);

protected:
  bool Simulate2D;           ///<Activates or deactivates 2D simulation (Y-forces are cancelled).
  bool Cpu;                  ///<Activates or deactivates only CPU simulation.
  bool Stable;               ///<Ensures the same results when repeated the same simulation.
  std::string AppName;
  std::string Hardware;
  std::string RunCode,RunTimeDate;

  TpCellOrder CellOrder;     ///<Direction (X,Y,Z) used to order particles.
  TpCellMode CellMode;       ///<Modes of cells division.
  unsigned Hdiv;             ///<Value to divide 2h.

  TpStep TStep;              ///<Type of step algorithm. 
  TpKernel TKernel;          ///<Type of kernel function. 
  StCubicCte CubicCte;       ///<Constants for Cubic Spline Kernel.
  StWendlandCte WendlandCte; ///<Constants for Wendland Kernel.
  TpVisco TVisco;            ///<Type of viscosity treatment.

  //multiphase
  StPhaseCte* PhaseCte;
  StPhaseArray* PhaseArray;
     
  TpDeltaSph TDeltaSph;      ///<Type of Delta-SPH approach: None o Dynamic.
  float DeltaSph;            ///<Constant for Delta-SPH. Default value is 0.1f (disabled with 0).

  JLog2 *Log;                ///<Declares an object of the class \ref JLog2.
  int VerletSteps;           ///<Verlet: number of steps to apply Eulerian equations (def=40).
  int ShepardSteps;          ///<Number of steps to apply Shepard density filter.
  std::string CaseName;      ///<Name of the case.
  std::string DirCase;       ///<Name of the input directory.
  std::string DirOut;        ///<Name of the output directory.
  std::string RunName;       ///<Name of the execution case.

  //-Variables to restart a simulation.
  std::string PartBeginDir;  ///<Directory with the PART file to start.
  unsigned PartBegin;        ///<Indicates the PART file to start.
  unsigned PartBeginFirst;   ///<Indicates the number of the firest PART file to be created.
  float PartBeginTimeStep;   ///<Instant of the beginning of the simulation.
  ullong PartBeginTotalNp;

  float H;                   ///<Smoothing length (=coef*sqrt(dx*dx+dy*dy+dz*dz)).
  float CteB;                ///<Constant that sets a limit for the maximum change in density.
  float Gamma;               ///<Politropic constant. (1-7).
  float Eps;                 ///<Epsilon constant for XSPH variant.
  float Visco;               ///<Viscosity value.
  float CFLnumber;           ///<Constant for the Courant condition (0.1 - 0.5).
  float Cs0;                 ///<Speed of sound of reference.
  float Delta2H;             ///<Constant for DeltaSPH. Delta2H=DeltaSph*H*2.
  float Dp;                  ///<Inter-particle distance.
  float MassBound;           ///<Mass of a boundary particle.
  float MassFluid;           ///<Mass of a fluid particle.
  float TimeMax;             ///<Time of simulation.
  float TimePart;            ///<Time between output files.
  float Rhop0;               ///<The value of Rhop0.
  float OverRhop0;           ///<The value of 1/Rhop0.

  byte SaveMode;
  tfloat3 Gravity;           ///<Gravity acceleration.
  float DtIni;               ///<Initial time step.
  float DtMin;               ///<Minimum time step.
  JSphDtFixed* DtFixed;      ///<Object that manages the use of fixed DT.

  float Dosh;                ///<2*\ref H.
  float H2;                  ///<\ref H*\ref H.
  float Fourh2;              ///<\ref H2*\ref H2.
  float Eta2;                ///<eta*eta being eta=0.1*\ref H
  float CteShepard;          ///<Constant used in Shepard filter to evaluate the self contribution of the particle itself.
  float SpsSmag;             ///<Smagorinsky constant used in SPS turbulence model.
  float SpsBlin;             ///<Blin constant used in the SPS turbulence model.
  unsigned PartOutMax;       ///<Allowed percentage of fluid particles out the domain.

  bool RhopOut;              ///<Indicates whether the density correction RhopOut is activated or not.
  float RhopOutMin;          ///<Minimum value for the density correction RhopOut.
  float RhopOutMax;          ///<Minimum value for the density correction RhopOut.

  byte SvData;               ///<Indicates the format of the output files. 
  bool SvDt;                 ///<Stores a file with info of DT.
  bool SvRes;                ///<Stores a file with info of the execution.  
  bool SvTimers;             ///<Computes the time for each process.
  
  bool SvSteps;              ///<Stores the output data of all time steps. 

  bool SvDomainVtk;          ///<Stores VTK file with the domain of particles of each PART file.

  tdouble3 CasePosMin,CasePosMax;  //-Limites de particulas del caso en instante inicial.       ///<Particle limits of the case in the initial instant.
  tdouble3 MapRealPosMin,MapRealPosMax,MapRealSize;
  unsigned CaseNp;           ///<Number of total particles.  
  unsigned CaseNfixed;       ///<Number of fixed boundary particles. 
  unsigned CaseNmoving;      ///<Number of moving boundary particles. 
  unsigned CaseNfloat;       ///<Number of floating boundary particles. 
  unsigned CaseNfluid;       ///<Number of fluid particles (including the excluded ones). 
  unsigned CaseNbound;       ///<Number of boundary particles ( \ref CaseNfixed + \ref CaseNmoving + \ref CaseNfloat ).
  unsigned CaseNpb;          ///<Number of particles of the boundary block ( \ref CaseNbound - \ref CaseNfloat ) or ( \ref CaseNfixed + \ref CaseNmoving).

  //Multiphase Particles Count/phase (most in the structure)
  unsigned Phases;
  

  StMkInfo *MkList;          ///<Amount of particles for each MK value.
  unsigned MkListSize;       ///<Number of different MK values.
  unsigned MkListFixed;      ///<Number of MK values of set of fixed particles.
  unsigned MkListMoving;     ///<Number of MK values of set of moving particles.
  unsigned MkListFloat;      ///<Number of MK values of set of floating particles.
  unsigned MkListBound;      ///<Number of MK values of set of boundary particles.
  unsigned MkListFluid;      ///<Number of MK values of set of fluid particles.

  tfloat3 MapPosMin;         ///<Minimum limits of the map for the case.
  tfloat3 MapPosMax;         ///<Maximum limits of the map for the case.
  tuint3 MapCells;           ///<Maximum number of cells according to the limits of the case. 

  byte PeriActive;           ///<Activate the use of periodic boundary conditions.
  bool PeriX;                ///<Periodic boundaries are used in X-direction.
  bool PeriY;                ///<Periodic boundaries are used in Y-direction.
  bool PeriZ;                ///<Periodic boundaries are used in Z-direction.
  bool PeriXY;               ///<Periodic boundaries are used in XY-direction.
  bool PeriXZ;               ///<Periodic boundaries are used in XZ-direction.
  bool PeriYZ;               ///<Periodic boundaries are used in YZ-direction.
  tfloat3 PeriXinc;          ///<Value to be added to the last position in X.
  tfloat3 PeriYinc;          ///<Value to be added to the last position in Y.
  tfloat3 PeriZinc;          ///<Value to be added to the last position in Z.

   //-Controls particle number.
  bool NpDynamic;   ///<CaseNp can increase.
  bool ReuseIds;    ///<Id of particles excluded values ​​are reused.
  ullong TotalNp;   ///<Total number of simulated particles (no cuenta las particulas inlet no validas).
  unsigned IdMax;   ///<It is the maximum Id used.

  unsigned DtModif;          ///<Number of times that DT was modified when it is too low.

  long long MaxMemoryCpu;    ///<Amount of allocated CPU memory.
  long long MaxMemoryGpu;    ///<Amount of allocated CPU memory.
  unsigned MaxParticles;     ///<Maximum number of particles.
  unsigned MaxCells;         ///<Maximum number of cells.

  int PartIni;               ///<First PART file.
  int Part;                  ///<Next PART to be stored.
  int Nstep;                 ///<Current time step.
  int PartNstep;             ///<Number of steps when the last PART was stored.    
  float PartDtMin;           ///<Minimum DT used on the last PART file.
  float PartDtMax;           ///<Maximum DT used on the last PART file.
  unsigned PartOut;          ///<Number of particles out (excluded) when the last PART was stored.
  float TimeStepIni;         ///<Initial instant of the simulation.
  float TimeStep;            ///<Current instant of the simulation.
  float TimeStepM1;          ///<Instant of the simulation when the last PART was stored.
  JTimer TimerTot;           ///<Total runtime of the simulation.
  JTimer TimerSim;           ///<Total runtime starting when the computation of the main loop starts.
  JTimer TimerPart;          ///<Runtime since the last PART to the next one.

  JSphMotion* Motion;        ///<Declares an object of the class \ref JSphMotion.
  float MotionTimeMod;       ///<Modifier of \ref TimeStep for \ref Motion.
  unsigned MotionObjCount;   ///<Number of objects with motion.
  unsigned MotionObjBegin[256];

  StFloatingData* FtObjs;    ///<Data of floating object.
  unsigned FtCount;          ///<Number of floating objects.
  float FtPause;             ///<Time to start floating bodies movement.
  std::string FtFileData;    ///<Name of file with info of floating bodies.
  JFloatingData* FtData;     ///<Info of floating bodies.
  
  JSphVarAcc *VarAcc;        ///<Object for variable acceleration functionality.

  void AllocMemoryFloating(unsigned ftcount);
  long long GetAllocMemoryCpu()const;

  void InitMultiPhase();	// Initialize multiphase GF
  
  void LoadConfig(const JCfgRun *cfg);
  void LoadCaseConfig();
  void ResetMkInfo();
  void LoadMkInfo(const JSpaceParts *parts);
  inline unsigned GetPosMkInfo(unsigned id)const;
  inline word CodeSetType(word code,TpParticle type,unsigned value)const;

  inline unsigned GetMkBlockById(unsigned id)const;
  unsigned GetMkBlockByMk(word mk)const;

  //void LoadCodeParticles(unsigned np,unsigned nout,const unsigned *idp,word *code)const;
  void LoadCodeParticles(unsigned np,const unsigned *idp,word *code)const;
  
  void ResizeMapLimits();

  void ConfigConstants(bool simulate2d);
  void VisuConfig()const;

  //void ConfigCellOrder(TpCellOrder order,unsigned np,tdouble3* pos,tfloat4* velrhop);
  void ConfigCellOrder(TpCellOrder order,unsigned np,tfloat3* pos,tfloat3* vel);
  void DecodeCellOrder(unsigned np,tdouble3 *pos,tfloat3 *vel)const;
  tuint3 OrderCode(const tuint3 &v)const{ return(OrderCodeValue(CellOrder,v)); }

  tfloat3 OrderCode(const tfloat3 &v)const{ return(OrderCodeValue(CellOrder,v)); }
  tfloat3 OrderDecode(const tfloat3 &v)const{ return(OrderDecodeValue(CellOrder,v)); }

  tdouble3 OrderCode(const tdouble3 &v)const{ return(OrderCodeValue(CellOrder,v)); }
  tdouble3 OrderDecode(const tdouble3 &v)const{ return(OrderDecodeValue(CellOrder,v)); }

  tuint3 OrderDecode(const tuint3 &v)const{ return(OrderDecodeValue(CellOrder,v)); }
  
  tmatrix4f OrderCode(const tmatrix4f &v)const{ return(OrderCodeValue(CellOrder,v)); }
  tmatrix4d OrderCode(const tmatrix4d &v)const{ return(OrderCodeValue(CellOrder,v)); }

  static void OrderCodeData(TpCellOrder order,unsigned n,tfloat3 *v);
  static void OrderDecodeData(TpCellOrder order,unsigned n,tfloat3 *v){ OrderCodeData(GetDecodeOrder(order),n,v); }
  static void OrderCodeData(TpCellOrder order,unsigned n,tdouble3 *v);
  static void OrderDecodeData(TpCellOrder order,unsigned n,tdouble3 *v){ OrderCodeData(GetDecodeOrder(order),n,v); }
  static void OrderCodeData(TpCellOrder order,unsigned n,tfloat4 *v);
  static void OrderDecodeData(TpCellOrder order,unsigned n,tfloat4 *v){ OrderCodeData(GetDecodeOrder(order),n,v); }

/*  void ConfigCellOrder(TpCellOrder order,unsigned np,tfloat3* pos,tfloat3* vel);
  tuint3 OrderCode(const tuint3 &v)const{ return(OrderCodeValue(CellOrder,v)); }
  tfloat3 OrderCode(const tfloat3 &v)const{ return(OrderCodeValue(CellOrder,v)); }
  tfloat3 OrderDecode(const tfloat3 &v)const{ return(OrderDecodeValue(CellOrder,v)); }
  tuint3 OrderDecode(const tuint3 &v)const{ return(OrderDecodeValue(CellOrder,v)); }
  tmatrix4f OrderCode(const tmatrix4f &v)const{ return(OrderCodeValue(CellOrder,v)); }
  static void OrderCodeData(TpCellOrder order,unsigned n,tfloat3 *v);
  static void OrderDecodeData(TpCellOrder order,unsigned n,tfloat3 *v){ OrderCodeData(GetDecodeOrder(order),n,v); } */

  void PrintSizeNp(unsigned np,long long size)const;
  void PrintHeadPart();

  void InitFloatingData();
  void SaveFloatingData();
  void SaveFloatingDataTotal();

  void ConfigSaveData(unsigned piece,unsigned pieces,std::string div);
  void AddParticlesOut(unsigned nout,const unsigned *idp,const tfloat3* pos,const tfloat3 *vel,const float *rhop,unsigned noutrhop,unsigned noutmove);

  
  void SaveBinData(const char *suffixpart,unsigned npok,unsigned nout,const unsigned *idp,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float*viscop,const tsymatrix3f *vtau,const unsigned *idpm);
  //void SaveData(unsigned npok,unsigned noutpos,unsigned noutrhop,unsigned noutmove,const unsigned *idp,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float*viscop,const tsymatrix3f *vtau,const unsigned *idpm,unsigned ndom,const tfloat3 *vdom);
  
  void SaveData(unsigned npok,const unsigned *idp,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float*viscop,const tsymatrix3f *vtau,const unsigned *idpm,unsigned ndom,const tdouble3 *vdom,const StInfoPartPlus *infoplus);

  void SavePartData(unsigned npok,unsigned nout,const unsigned *idp,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float*viscop,const tsymatrix3f *vtau,const unsigned *idpm,unsigned ndom,const tdouble3 *vdom,const StInfoPartPlus *infoplus);

  void SaveDomainVtk(unsigned ndom,const tdouble3 *vdom)const;
  void SaveMapCellsVtk(float scell)const;

  void GetResInfo(float tsim,float ttot,const std::string &headplus,const std::string &detplus,std::string &hinfo,std::string &dinfo);
  void SaveRes(float tsim,float ttot,const std::string &headplus="",const std::string &detplus="");
  void ShowResume(bool stop,float tsim,float ttot,bool all,std::string infoplus);

  TpParticle GetTpParticleCode(word code)const{ const word type=CODE_GetType(code); return(type==CODE_TYPE_FLUID? PART_Fluid: (type==CODE_TYPE_FIXED? PART_BoundFx: (type==CODE_TYPE_MOVING? PART_BoundMv: PART_BoundFt))); }
  unsigned GetOutPosCount()const{ return(OutPosCount); }
  unsigned GetOutRhopCount()const{ return(OutRhopCount); }
  unsigned GetOutMoveCount()const{ return(OutMoveCount); }
  unsigned GetOutInitialCount()const{ return(OutInitialCount); }
  void SetOutInitialCount(unsigned nout){ OutInitialCount=nout; }

public:
  char Cad[1024];

  JSph();
  ~JSph();

  static std::string GetStepName(TpStep tstep);
  static std::string GetKernelName(TpKernel tkernel);
  static std::string GetViscoName(TpVisco tvisco);
  static std::string GetDeltaSphName(TpDeltaSph tdelta);

  static std::string TimerToText(const std::string &name,float value);

  //------Debug-----
public:
  //void DgSaveVtkParticlesCpu(std::string filename,int numfile,unsigned pini,unsigned pfin,const tfloat3 *pos,const byte *check=NULL,const unsigned *idp=NULL,const tfloat3 *vel=NULL,const float *rhop=NULL);
  //void DgSaveCsvParticlesCpu(std::string filename,int numfile,unsigned pini,unsigned pfin,std::string head,const tfloat3 *pos,const unsigned *idp=NULL,const tfloat3 *vel=NULL,const float *rhop=NULL,const float *ar=NULL,const tfloat3 *ace=NULL,const tfloat3 *vcorr=NULL);

};


#endif


