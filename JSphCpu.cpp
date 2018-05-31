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

/// \file JSphCpu.cpp \brief Implements the class \ref JSphCpu.

#include <climits>
#include "JSphCpu.h"
#include "Functions.h"
#include "JSphMotion.h"
#include "JCellDivCpu.h"
#include "JPartsOut.h"
#include "JSphDtFixed.h"
#include "JPeriodicCpu.h"
#include "JFloatingData.h"
#include "JSphVarAcc.h"

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JSphCpu::JSphCpu(){
  ClassName="JSphCpu";
  Cpu=true;
  Idp=NULL; Code=NULL; Pos=NULL; Vel=NULL; Rhop=NULL;
  Ace=NULL; Ar=NULL; VelXcor=NULL; Delta=NULL; 
  Csound=NULL; PrRhop=NULL; Tensil=NULL;
  VelM1=NULL; RhopM1=NULL;                  //-Verlet.
  PosPre=NULL; VelPre=NULL; RhopPre=NULL;   //-Symplectic.
  FdWab=NULL; FdRhop=NULL;                  //-Shepard.
  Tau=NULL; Csph=NULL;                      //-Laminar+SPS. 
  CellDiv=NULL; PeriodicZone=NULL;
  PartsOut=new JPartsOut;
  RidpMoving=NULL; 
  FtRidp=NULL; FtDist=NULL;
  InitVars();
  TmcCreation(Timers,false);
}

//==============================================================================
/// Destructor.
//==============================================================================
JSphCpu::~JSphCpu(){
  FreeMemoryParticles();
  FreeMemoryAuxParticles();
  delete PartsOut;
  TmcDestruction(Timers);
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSphCpu::InitVars(){
  RunMode="";
  Scell=0;
  OmpThreads=1;
  OmpMode=OMPM_Single;
  Np=Npb=NpbOk=0;
  WithFloating=false;
  FreeMemoryParticles();
  FreeMemoryAuxParticles();
  CellDiv=NULL; PeriodicZone=NULL;
  Idpi=Idpj=NULL; Codei=Codej=NULL; Posi=Posj=NULL; Veli=Velj=NULL; Rhopi=Rhopj=NULL;
  PrRhopi=PrRhopj=NULL; Csoundi=Csoundj=NULL; Tensili=Tensilj=NULL; Taui=Tauj=NULL;
}

//==============================================================================
/// Releases memory of main data.
//==============================================================================
void JSphCpu::FreeMemoryParticles(){
  ParticlesSize=0;
  MemCpuParticles=0;
  delete[] Idp;      Idp=NULL;
  delete[] Code;     Code=NULL;
  delete[] Pos;      Pos=NULL;
  delete[] Vel;      Vel=NULL;
  delete[] Rhop;     Rhop=NULL;
  delete[] VelM1;   VelM1=NULL;   //-Verlet
  delete[] RhopM1;  RhopM1=NULL;  //-Verlet
  delete[] PosPre;  PosPre=NULL;  //-Symplectic
  delete[] VelPre;  VelPre=NULL;  //-Symplectic
  delete[] RhopPre; RhopPre=NULL; //-Symplectic
  delete[] Tau;     Tau=NULL;     //-Laminar+SPS   
  delete[] RidpMoving; RidpMoving=NULL; 
  delete[] FtRidp;     FtRidp=NULL; 
  delete[] FtDist;     FtDist=NULL;
}

//==============================================================================
/// Allocates memory of main data.
//==============================================================================
void JSphCpu::AllocMemoryParticles(unsigned np){
  const char* met="AllocMemoryParticles";
  FreeMemoryParticles();
  ParticlesSize=np;
  if(np>0){
    try{
      Idp=new unsigned[np];      MemCpuParticles+=sizeof(unsigned)*np;
      Code=new word[np];         MemCpuParticles+=sizeof(word)*np;
      Pos=new tfloat3[np];       MemCpuParticles+=sizeof(tfloat3)*np;
      Vel=new tfloat3[np];       MemCpuParticles+=sizeof(tfloat3)*np;
      Rhop=new float[np];        MemCpuParticles+=sizeof(float)*np;
      if(TStep==STEP_Verlet){
        VelM1=new tfloat3[np];   MemCpuParticles+=sizeof(tfloat3)*np;
        RhopM1=new float[np];    MemCpuParticles+=sizeof(float)*np; 
      } 
      if(TStep==STEP_Symplectic){
        PosPre=new tfloat3[np];   MemCpuParticles+=sizeof(tfloat3)*np;
        VelPre=new tfloat3[np];   MemCpuParticles+=sizeof(tfloat3)*np;
        RhopPre=new float[np];    MemCpuParticles+=sizeof(float)*np; 
      } 
      if(TVisco==VISCO_LaminarSPS){   
        Tau=new tsymatrix3f[np];  MemCpuParticles+=sizeof(tsymatrix3f)*np;
      }

      //-Allocates memory for arrays with fixed size.
      if(CaseNmoving){
        RidpMoving=new unsigned[CaseNmoving];  MemCpuParticles+=sizeof(unsigned)*CaseNmoving;
      }
      if(CaseNfloat){
        FtRidp=new unsigned[CaseNfloat];     MemCpuParticles+=sizeof(unsigned)*CaseNfloat;
        FtDist=new tfloat3[CaseNfloat];      MemCpuParticles+=sizeof(tfloat3)*CaseNfloat;
      }

    }
    catch(const std::bad_alloc){
      RunException(met,"The requested memory could not be allocated.");
    }
  }
}

//==============================================================================
/// Releases memory of auxiliar/volatile data.
//==============================================================================
void JSphCpu::FreeMemoryAuxParticles(){
  delete[] Ace;       Ace=NULL;
  delete[] Ar;        Ar=NULL;
  delete[] VelXcor;   VelXcor=NULL;
  delete[] Delta;     Delta=NULL;
  delete[] Csound;    Csound=NULL;
  delete[] PrRhop;    PrRhop=NULL;
  delete[] Tensil;    Tensil=NULL;
  delete[] FdWab;     FdWab=NULL;
  delete[] FdRhop;    FdRhop=NULL;
  delete[] Csph;      Csph=NULL; 
}

//==============================================================================
/// Allocates memory of auxiliar/volatile data.
//==============================================================================
void JSphCpu::AllocMemoryAuxParticles(unsigned np,unsigned npf){
  const char* met="AllocMemoryAuxParticles";
  FreeMemoryAuxParticles();
  if(np>0){
    const unsigned npmul=(OmpMode!=OMPM_Single? OmpThreads: 1);
    try{
      const unsigned npite=np*npmul;
      Ace=new tfloat3[npite];        MemCpuParticles+=sizeof(tfloat3)*npite;
      Ar=new float[npite];           MemCpuParticles+=sizeof(float)*npite;
      VelXcor=new tfloat3[npite];    MemCpuParticles+=sizeof(tfloat3)*npite;
      if(TDeltaSph==DELTA_DBCExt){
        Delta=new float[npite];        MemCpuParticles+=sizeof(float)*npite;
      }
      Csound=new float[np];          MemCpuParticles+=sizeof(float)*np;
      PrRhop=new float[np];          MemCpuParticles+=sizeof(float)*np;
      if(TKernel==KERNEL_Cubic){
        Tensil=new float[np];        MemCpuParticles+=sizeof(float)*np;
      } 
      if(ShepardSteps){
        const unsigned npfite=npf*npmul;
        FdWab=new float[npfite];      MemCpuParticles+=sizeof(float)*npfite;
        FdRhop=new float[npfite];     MemCpuParticles+=sizeof(float)*npfite;
      }
      if(TVisco==VISCO_LaminarSPS){   
        const unsigned npfomp=npf*npmul;
        Csph=new tsymatrix3f[npfomp];  MemCpuParticles+=sizeof(tsymatrix3f)*npfomp;  
      }
    }
    catch(const std::bad_alloc){
      RunException(met,"The requested memory could not be allocated.");
    }
  }
}

//==============================================================================
/// Returns the allocated memory in CPU.
//==============================================================================
long long JSphCpu::GetAllocMemoryCpu()const{  
  long long s=JSph::GetAllocMemoryCpu();
  //Allocated in AllocMemoryParticles() & AllocMemoryAuxParticles()
  s+=MemCpuParticles;
  //Allocated in other objects.
  if(PartsOut)s+=PartsOut->GetAllocMemory();
  return(s);
}

//==============================================================================
/// Displays the allocated memory.
//==============================================================================
void JSphCpu::PrintAllocMemory(long long mcpu)const{
  char cad[128];
  sprintf(cad,"Allocated memory in CPU: %lld (%.2f MB)",mcpu,double(mcpu)/(1024*1024));
  Log->Print(cad);
}

//==============================================================================
/// Loads the configuration of execution with OpenMP.
//==============================================================================
void JSphCpu::ConfigOmp(const JCfgRun *cfg){
#ifdef _WITHOMP
  //-Determines the number of threads by host with OpenMP.
  if(Cpu&&cfg->OmpThreads!=1&&cfg->OmpMode!=OMPM_Single){
    OmpThreads=cfg->OmpThreads;
    if(OmpThreads<=0)OmpThreads=omp_get_num_procs();
    if(OmpThreads>MAXTHREADS_OMP)OmpThreads=MAXTHREADS_OMP;
    omp_set_num_threads(OmpThreads);
    Log->Print(string("Threads by host for parallel execution: ")+fun::IntStr(omp_get_max_threads()));
  }
  else{
    OmpThreads=1;
    omp_set_num_threads(OmpThreads);
  }
  OmpMode=cfg->OmpMode;
  if(OmpMode==OMPM_Dynamic&&cfg->Stable){
    Log->Print("*** The OMP mode Dynamic is changed to Static because the Stable option is used.");
    OmpMode=OMPM_Static;
  }
#else
  OmpThreads=1;
  OmpMode=OMPM_Single;
#endif
}

//==============================================================================
/// Configures mode of CPU execution.
//==============================================================================
void JSphCpu::ConfigRunMode(const JCfgRun *cfg,const std::string &preinfo){
  Hardware="Cpu";
  if(OmpMode==OMPM_Single)RunMode=string("Single core, Symmetry:")+(USE_SYMMETRY? "True": "False");
  else{
    RunMode=string("OpenMP(Threads:")+fun::IntStr(OmpThreads);
    RunMode=RunMode+", mode:";
    if(OmpMode==OMPM_Dynamic)RunMode=RunMode+"Dynamic";
    if(OmpMode==OMPM_Static)RunMode=RunMode+"Static";
    RunMode=RunMode+")";
  }
  if(!preinfo.empty())RunMode=preinfo+", "+RunMode;
  if(Stable)RunMode=string("Stable, ")+RunMode;
  Log->Print(fun::VarStr("RunMode",RunMode));
}

//==============================================================================
/// Computes for a range of particles their position according to segun Idp[].
//==============================================================================
void JSphCpu::CalcRidp(unsigned n,unsigned ini,unsigned idini,unsigned idfin,const unsigned *idp,unsigned *ridp)const{
  memset(ridp,255,sizeof(unsigned)*(idfin-idini)); //-Assigns UINT_MAX values.
  for(unsigned p=0;p<n;p++){
    unsigned id=idp[p+ini];
    if(idini<=id&&id<idfin)ridp[id-idini]=p+ini;
  }
}

//==============================================================================
/// Adjusts variables of particles of floating bodies.dy
//==============================================================================
void JSphCpu::InitFloating(){
  JSph::InitFloatingData();
  if(!PartBegin){
    //-Gets positions of floating particles.
    CalcFtRidp();
    //-Computes distance of particles to center of the object.
    for(unsigned cf=0;cf<FtCount;cf++){
      const StFloatingData* fobj=FtObjs+cf;
      const unsigned ini=fobj->begin-CaseNpb;
      const unsigned fin=ini+fobj->count;
      for(unsigned fp=ini;fp<fin;fp++){
        int p=FtRidp[fp];
        FtDist[fp]=TFloat3(Pos[p].x-fobj->center.x,Pos[p].y-fobj->center.y,Pos[p].z-fobj->center.z);
      }
    }
    //-Stores distance of particles to center of the object.
    if(CellOrder!=ORDER_XYZ)OrderDecodeData(CellOrder,CaseNfloat,FtDist);
    FtData->AddDist(CaseNfloat,FtDist);
    if(CellOrder!=ORDER_XYZ)OrderCodeData(CellOrder,CaseNfloat,FtDist);
  }
  else{
    //-Recovers distance of particles to center of the object.
    memcpy(FtDist,FtData->GetDist(CaseNfloat),sizeof(tfloat3)*CaseNfloat);
    if(CellOrder!=ORDER_XYZ)OrderCodeData(CellOrder,CaseNfloat,FtDist);
  }
}

//==============================================================================
/// Initialises arrays and variables for the CPU execution.
//==============================================================================
void JSphCpu::InitRun(){
  if(TStep==STEP_Verlet){
    memcpy(VelM1,Vel,sizeof(tfloat3)*Np);
    memcpy(RhopM1,Rhop,sizeof(float)*Np);
    VerletStep=0;
  }
  else if(TStep==STEP_Symplectic){
    DtPre=DtIni;
  }
  if(TVisco==VISCO_LaminarSPS){   
    memset(Tau,0,sizeof(tsymatrix3f)*Np);
  }
  WithFloating=(CaseNfloat>0);
  if(CaseNfloat)InitFloating();

  //-Adjusts paramaters to start.
  PartIni=PartBeginFirst;
  TimeStepIni=(!PartIni? 0: PartBeginTimeStep);
  //-Adjusts motion for the instant of the loaded PART.
  if(CaseNmoving){
    MotionTimeMod=(!PartIni? PartBeginTimeStep: 0);
    Motion->ProcesTime(0,TimeStepIni+MotionTimeMod);
  }

  Part=PartIni; Nstep=0; PartNstep=0; PartOut=0;
  TimeStep=TimeStepIni; TimeStepM1=TimeStep;
  if(DtFixed)DtIni=DtFixed->GetDt(TimeStep,DtIni);
}

//==============================================================================
/// Adds variable acceleration from input files.
//==============================================================================
void JSphCpu::AddVarAcc(){
  for(unsigned c=0;c<VarAcc->GetCount();c++){
    unsigned mkvalue;
    tfloat3 acclin,accang,centre;
    VarAcc->GetAccValues(c,TimeStep,mkvalue,acclin,accang,centre);
    const bool withaccang=(accang.x!=0||accang.y!=0||accang.z!=0);
    const word codesel=word(CODE_TYPE_FLUID|mkvalue);

    const int npb=int(Npb),np=int(Np);
    #ifdef _WITHOMP
      #pragma omp parallel for schedule (static)
    #endif
    for(int p=npb;p<np;p++){                //Iterates through the fluid particles.
                                            //Checks if the current particle is part of the particle set by its MK.
      if(CODE_GetValue(Code[p])==codesel){
        tfloat3 acc=acclin;                 //Adds linear acceleration.
        if(withaccang){                     //Adds angular acceleration.
          const tfloat3 dc=Pos[p]-centre;
          acc.x+=accang.y*dc.z-accang.z*dc.y;
          acc.y+=accang.z*dc.x-accang.x*dc.z;
          acc.z+=accang.x*dc.y-accang.y*dc.x;
        }
        Ace[p]=Ace[p]+acc;
      }
    }
  }
}

//==============================================================================
/// Prepares variables for interaction "INTER_Forces" or "INTER_ForcesCorr".
//==============================================================================
void JSphCpu::PreInteraction_Forces(TpInter tinter){
  switch(TKernel){
    case KERNEL_Cubic:     PreInteraction_Forces_<KERNEL_Cubic>(tinter);     break;
    case KERNEL_Wendland:  PreInteraction_Forces_<KERNEL_Wendland>(tinter);  break;
  }
}
//==============================================================================
template<TpKernel tker> void JSphCpu::PreInteraction_Forces_(TpInter tinter){
  TmcStart(Timers,TMC_CfPreForces);
  if(OmpMode!=OMPM_Single){
    int npite=Np*OmpThreads;
    memset(Ar,0,sizeof(float)*npite);                                        //Ar[]=0.
    if(tinter==INTER_Forces)memset(VelXcor,0,sizeof(tfloat3)*npite);         //VelXcor[]=(0,0,0).
    if(Delta)memset(Delta,0,sizeof(float)*npite);                            //Delta[]=(0,0,0).
    if(TVisco==VISCO_LaminarSPS)memset(Csph,0,sizeof(tsymatrix3f)*((Np-Npb)*OmpThreads));  //Csph[]=(0,0,0,0,0,0).
    memset(Ace,0,sizeof(tfloat3)*npite);                                     //Ace[]=0 (although only interval npb:np is used).
  }
  else{
    memset(Ar,0,sizeof(float)*Np);                                           //Ar[]=0.
    if(tinter==INTER_Forces)memset(VelXcor+Npb,0,sizeof(tfloat3)*(Np-Npb));  //VelXcor[]=(0,0,0).
    if(Delta)memset(Delta+Npb,0,sizeof(float)*(Np-Npb));                     //Delta[]=(0,0,0).
    if(TVisco==VISCO_LaminarSPS)memset(Csph,0,sizeof(tsymatrix3f)*(Np-Npb)); //Csph[]=(0,0,0,0,0,0).
    memset(Ace,0,sizeof(tfloat3)*Np);                                        //Ace[]=0.
  }

  for(unsigned p=Npb;p<Np;p++)Ace[p]=Gravity;                                //Ace[]=Gravity.

  //There are variable acceleration input files.
  if(VarAcc)AddVarAcc();

  const int np=int(Np);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (static)
  #endif
  for(int p=0;p<np;p++){
    float rhop=Rhop[p],rhop_r0=rhop*OverRhop0; 
    Csound[p]=Cs0*(rhop_r0*rhop_r0*rhop_r0);
    float press=CteB*(pow(rhop_r0,Gamma)-1.0f);
    PrRhop[p]=press/(rhop*rhop);
    if(tker==KERNEL_Cubic)Tensil[p]=PrRhop[p]*(press>0? 0.01f: -0.2f);
  }
  CsoundMax=0;
#ifdef DT_ALLPARTICLES
  for(unsigned p=0;p<Np;p++)CsoundMax=max(CsoundMax,Csound[p]);
#else
  for(unsigned p=Npb;p<Np;p++)CsoundMax=max(CsoundMax,Csound[p]);
#endif
  ViscDtMax=0;
  for(int t=0;t<OmpThreads;t++)ViscDtThread[t*STRIDE_OMP]=0;
  TmcStop(Timers,TMC_CfPreForces);
}

//==============================================================================
/// Prepares variables for interaction "INTER_Shepard".
//==============================================================================
void JSphCpu::PreInteraction_Shepard(){
  unsigned sshepard=(Np-Npb)*sizeof(float)*OmpThreads;
  memset(FdWab,0,sshepard);  //FdWab[]=0.
  memset(FdRhop,0,sshepard); //FdRhop[]=0.
}

//==============================================================================
/// Selection of parameters template for InteractionCells.
//==============================================================================
void JSphCpu::InteractionCells(TpInter tinter){
  ConfigComputeDataij(Idp,Code,Pos,Rhop,Vel,PrRhop,Csound,Tensil,Tau);
  if(OmpMode==OMPM_Single){
    if(tinter==INTER_Forces){ const TpInter tinte=INTER_Forces;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Single<tinte,tker,tvis,true>(); else InteractionCells_Single<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Single<tinte,tker,tvis,true>(); else InteractionCells_Single<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Single<tinte,tker,tvis,true>(); else InteractionCells_Single<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Single<tinte,tker,tvis,true>(); else InteractionCells_Single<tinte,tker,tvis,false>();
        }
      }
    }else if(tinter==INTER_ForcesCorr){ const TpInter tinte=INTER_ForcesCorr;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Single<tinte,tker,tvis,true>(); else InteractionCells_Single<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Single<tinte,tker,tvis,true>(); else InteractionCells_Single<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Single<tinte,tker,tvis,true>(); else InteractionCells_Single<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS;
          if(WithFloating)InteractionCells_Single<tinte,tker,tvis,true>(); else InteractionCells_Single<tinte,tker,tvis,false>();
        }
      }
    }else if(tinter==INTER_Shepard){ const TpInter tinte=INTER_Shepard;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Single<tinte,tker,tvis,true>(); else InteractionCells_Single<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Single<tinte,tker,tvis,true>(); else InteractionCells_Single<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Single<tinte,tker,tvis,true>(); else InteractionCells_Single<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Single<tinte,tker,tvis,true>(); else InteractionCells_Single<tinte,tker,tvis,false>();
        }
      }
    }
  }
#ifdef _WITHOMP
  else if(OmpMode==OMPM_Static){
    if(tinter==INTER_Forces){ const TpInter tinte=INTER_Forces;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Static<tinte,tker,tvis,true>(); else InteractionCells_Static<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Static<tinte,tker,tvis,true>(); else InteractionCells_Static<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Static<tinte,tker,tvis,true>(); else InteractionCells_Static<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Static<tinte,tker,tvis,true>(); else InteractionCells_Static<tinte,tker,tvis,false>();
        }
      }
    }else if(tinter==INTER_ForcesCorr){ const TpInter tinte=INTER_ForcesCorr;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Static<tinte,tker,tvis,true>(); else InteractionCells_Static<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Static<tinte,tker,tvis,true>(); else InteractionCells_Static<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Static<tinte,tker,tvis,true>(); else InteractionCells_Static<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Static<tinte,tker,tvis,true>(); else InteractionCells_Static<tinte,tker,tvis,false>();
        }
      }
    }else if(tinter==INTER_Shepard){ const TpInter tinte=INTER_Shepard;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Static<tinte,tker,tvis,true>(); else InteractionCells_Static<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Static<tinte,tker,tvis,true>(); else InteractionCells_Static<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Static<tinte,tker,tvis,true>(); else InteractionCells_Static<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Static<tinte,tker,tvis,true>(); else InteractionCells_Static<tinte,tker,tvis,false>();
        }
      }
    }
  }
  else if(OmpMode==OMPM_Dynamic){
    if(tinter==INTER_Forces){ const TpInter tinte=INTER_Forces;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Dynamic<tinte,tker,tvis,true>(); else InteractionCells_Dynamic<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS;
          if(WithFloating)InteractionCells_Dynamic<tinte,tker,tvis,true>(); else InteractionCells_Dynamic<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Dynamic<tinte,tker,tvis,true>(); else InteractionCells_Dynamic<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Dynamic<tinte,tker,tvis,true>(); else InteractionCells_Dynamic<tinte,tker,tvis,false>();
        }
      }
    }else if(tinter==INTER_ForcesCorr){ const TpInter tinte=INTER_ForcesCorr;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Dynamic<tinte,tker,tvis,true>(); else InteractionCells_Dynamic<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Dynamic<tinte,tker,tvis,true>(); else InteractionCells_Dynamic<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Dynamic<tinte,tker,tvis,true>(); else InteractionCells_Dynamic<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Dynamic<tinte,tker,tvis,true>(); else InteractionCells_Dynamic<tinte,tker,tvis,false>();
        }
      }
    }else if(tinter==INTER_Shepard){ const TpInter tinte=INTER_Shepard;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Dynamic<tinte,tker,tvis,true>(); else InteractionCells_Dynamic<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Dynamic<tinte,tker,tvis,true>(); else InteractionCells_Dynamic<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionCells_Dynamic<tinte,tker,tvis,true>(); else InteractionCells_Dynamic<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionCells_Dynamic<tinte,tker,tvis,true>(); else InteractionCells_Dynamic<tinte,tker,tvis,false>();
        }
      }
    }
  }
#endif
}

//==============================================================================
/// Cells interaction in Single-Core.
//==============================================================================
template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void JSphCpu::InteractionCells_Single(){
  const int ncx=int(CellDiv->GetNcx()),ncy=int(CellDiv->GetNcy()),ncz=int(CellDiv->GetNcz());
  const int nsheet=ncx*ncy,nct=nsheet*ncz;
  //-Shepard only with Fluid-Fluid.
  for(byte kind=(tinter==INTER_Shepard? 2: 1);kind<=2;kind++){ //-1:Boundary, 2:Fluid
    byte kind2ini=(tinter==INTER_Shepard? 2: 3-kind);          //(1->2, 2->1 & 2->2)
    for(int box=0;box<nct;box++){
      const int cz=int(box/nsheet);
      int bx=box-(cz*nsheet);
      const int cy=int(bx/ncx);
      const int cx=bx-(cy*ncx);
      if(CellDiv->CellNoEmpty(box,kind)){
        int ibegin=int(CellDiv->CellBegin(box,kind));
        int iend=int(CellDiv->CellBegin(box+1,kind));
        int xneg=-min(cx,int(Hdiv));
        int xpos=min(ncx-cx-1,int(Hdiv));
        int yneg=-min(cy,int(Hdiv));
        int ypos=min(ncy-cy-1,int(Hdiv));
        int zneg=-min(cz,int(Hdiv));
        int zpos=min(ncz-cz-1,int(Hdiv));
        InteractSelf<true,tinter,tker,tvis,floating>(box,kind,kind2ini); //Self
        if(cx+1<ncx)InteractCelij<true,tinter,tker,tvis,floating>(ibegin,iend,box+1,box+xpos,kind2ini); //East
        if(cy+1<ncy){ //North
          for(int y=1;y<=ypos;y++){
            const int box2=box+ncx*y;
            InteractCelij<true,tinter,tker,tvis,floating>(ibegin,iend,box2+xneg,box2+xpos,kind2ini);
          }
        }
        if(cz+1<ncz){ //Up
          for(int z=1;z<=zpos;z++)for(int y=yneg;y<=ypos;y++){
            const int box2=box+nsheet*z+ncx*y;
            InteractCelij<true,tinter,tker,tvis,floating>(ibegin,iend,box2+xneg,box2+xpos,kind2ini);
          }
        }
      }
    }
  }
}

#ifdef _WITHOMP
//==============================================================================
/// Cells interaction OpenMP Static.
//==============================================================================
template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void JSphCpu::InteractionCells_Static(){
  const int ncx=int(CellDiv->GetNcx()),ncy=int(CellDiv->GetNcy()),ncz=int(CellDiv->GetNcz());
  const int nsheet=ncx*ncy,nct=nsheet*ncz;
  //-Shepard only with Fluid-Fluid.
  for(byte kind=(tinter==INTER_Shepard? 2: 1);kind<=2;kind++){ //-1:Boundary, 2:Fluid
    byte kind2ini=(tinter==INTER_Shepard? 2: 3-kind);          //(1->2, 2->1 & 2->2)
    #pragma omp parallel for  schedule (static,10)
    for(int box=0;box<nct;box++){
      const int cz=int(box/nsheet);
      int bx=box-(cz*nsheet);
      const int cy=int(bx/ncx);
      const int cx=bx-(cy*ncx);
      if(CellDiv->CellNoEmpty(box,kind)){
        int ibegin=int(CellDiv->CellBegin(box,kind));
        int iend=int(CellDiv->CellBegin(box+1,kind));
        int xneg=-min(cx,int(Hdiv));
        int xpos=min(ncx-cx-1,int(Hdiv));
        int yneg=-min(cy,int(Hdiv));
        int ypos=min(ncy-cy-1,int(Hdiv));
        int zneg=-min(cz,int(Hdiv));
        int zpos=min(ncz-cz-1,int(Hdiv));
        InteractSelf<true,tinter,tker,tvis,floating>(box,kind,kind2ini);//Self
        if(cx+1<ncx)InteractCelij<true,tinter,tker,tvis,floating>(ibegin,iend,box+1,box+xpos,kind2ini); //East
        if(cy+1<ncy){ //North
          for(int y=1;y<=ypos;y++){
            const int box2=box+ncx*y;
            InteractCelij<true,tinter,tker,tvis,floating>(ibegin,iend,box2+xneg,box2+xpos,kind2ini);
          }
        }
        if(cz+1<ncz){ //Up
          for(int z=1;z<=zpos;z++)for(int y=yneg;y<=ypos;y++){
            const int box2=box+nsheet*z+ncx*y;
            InteractCelij<true,tinter,tker,tvis,floating>(ibegin,iend,box2+xneg,box2+xpos,kind2ini);
          }
        }
      }
    }
  }
}

//==============================================================================
/// Cells interaction OpenMP Dynamic.
//==============================================================================
template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void JSphCpu::InteractionCells_Dynamic(){
  const int ncx=int(CellDiv->GetNcx()),ncy=int(CellDiv->GetNcy()),ncz=int(CellDiv->GetNcz());
  const int nsheet=ncx*ncy,nct=nsheet*ncz;
  //-Shepard only with Fluid-Fluid.
  for(byte kind=(tinter==INTER_Shepard? 2: 1);kind<=2;kind++){ //-1:Boundary, 2:Fluid
    byte kind2ini=(tinter==INTER_Shepard? 2: 3-kind);          //(1->2, 2->1 & 2->2)
    #pragma omp parallel for schedule (dynamic,10)
    for(int box=0;box<nct;box++){
      const int cz=int(box/nsheet);
      int bx=box-(cz*nsheet);
      const int cy=int(bx/ncx);
      const int cx=bx-(cy*ncx);
      if(CellDiv->CellNoEmpty(box,kind)){
        int ibegin=int(CellDiv->CellBegin(box,kind));
        int iend=int(CellDiv->CellBegin(box+1,kind));
        int xneg=-min(cx,int(Hdiv));
        int xpos=min(ncx-cx-1,int(Hdiv));
        int yneg=-min(cy,int(Hdiv));
        int ypos=min(ncy-cy-1,int(Hdiv));
        int zneg=-min(cz,int(Hdiv));
        int zpos=min(ncz-cz-1,int(Hdiv));
        InteractSelf<true,tinter,tker,tvis,floating>(box,kind,kind2ini); //Self
        if(cx+1<ncx)InteractCelij<true,tinter,tker,tvis,floating>(ibegin,iend,box+1,box+xpos,kind2ini); //East
        if(cy+1<ncy){ //North
          for(int y=1;y<=ypos;y++){
            const int box2=box+ncx*y;
            InteractCelij<true,tinter,tker,tvis,floating>(ibegin,iend,box2+xneg,box2+xpos,kind2ini);
          }
        }
        if(cz+1<ncz){ //Up
          for(int z=1;z<=zpos;z++)for(int y=yneg;y<=ypos;y++){
            const int box2=box+nsheet*z+ncx*y;
            InteractCelij<true,tinter,tker,tvis,floating>(ibegin,iend,box2+xneg,box2+xpos,kind2ini);
          }
        }
      }
    }
  }
}
#endif

//==============================================================================
/// Interactions with periodic zone.
//==============================================================================
void JSphCpu::InteractionPeri(TpInter tinter,JPeriodicCpu* pzone){
  PeriodicZone=pzone;
  ConfigComputeDataij(PeriodicZone->GetIdp(),PeriodicZone->GetCode(),PeriodicZone->GetPos(),PeriodicZone->GetRhop(),PeriodicZone->GetVel(),PeriodicZone->GetPrRhop(),PeriodicZone->GetCsound(),PeriodicZone->GetTensil(),PeriodicZone->GetTau());
  if(OmpMode==OMPM_Single){
    if(tinter==INTER_Forces){ const TpInter tinte=INTER_Forces;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Single<tinte,tker,tvis,true>(); else InteractionPeri_Single<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Single<tinte,tker,tvis,true>(); else InteractionPeri_Single<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Single<tinte,tker,tvis,true>(); else InteractionPeri_Single<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Single<tinte,tker,tvis,true>(); else InteractionPeri_Single<tinte,tker,tvis,false>();
        }
      }
    }else if(tinter==INTER_ForcesCorr){ const TpInter tinte=INTER_ForcesCorr;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Single<tinte,tker,tvis,true>(); else InteractionPeri_Single<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Single<tinte,tker,tvis,true>(); else InteractionPeri_Single<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Single<tinte,tker,tvis,true>(); else InteractionPeri_Single<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Single<tinte,tker,tvis,true>(); else InteractionPeri_Single<tinte,tker,tvis,false>();
        }
      }
    }else if(tinter==INTER_Shepard){ const TpInter tinte=INTER_Shepard;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Single<tinte,tker,tvis,true>(); else InteractionPeri_Single<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Single<tinte,tker,tvis,true>(); else InteractionPeri_Single<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Single<tinte,tker,tvis,true>(); else InteractionPeri_Single<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Single<tinte,tker,tvis,true>(); else InteractionPeri_Single<tinte,tker,tvis,false>();
        }
      }
    }
  }
  else if(OmpMode==OMPM_Static){
    if(tinter==INTER_Forces){ const TpInter tinte=INTER_Forces;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Static<tinte,tker,tvis,true>(); else InteractionPeri_Static<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Static<tinte,tker,tvis,true>(); else InteractionPeri_Static<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Static<tinte,tker,tvis,true>(); else InteractionPeri_Static<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Static<tinte,tker,tvis,true>(); else InteractionPeri_Static<tinte,tker,tvis,false>();
        }
      }
    }else if(tinter==INTER_ForcesCorr){ const TpInter tinte=INTER_ForcesCorr;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Static<tinte,tker,tvis,true>(); else InteractionPeri_Static<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Static<tinte,tker,tvis,true>(); else InteractionPeri_Static<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Static<tinte,tker,tvis,true>(); else InteractionPeri_Static<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Static<tinte,tker,tvis,true>(); else InteractionPeri_Static<tinte,tker,tvis,false>();
        }
      }
    }else if(tinter==INTER_Shepard){ const TpInter tinte=INTER_Shepard;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Static<tinte,tker,tvis,true>(); else InteractionPeri_Static<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Static<tinte,tker,tvis,true>(); else InteractionPeri_Static<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Static<tinte,tker,tvis,true>(); else InteractionPeri_Static<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Static<tinte,tker,tvis,true>(); else InteractionPeri_Static<tinte,tker,tvis,false>();
        }
      }
    }
  }
  else if(OmpMode==OMPM_Dynamic){
    if(tinter==INTER_Forces){ const TpInter tinte=INTER_Forces;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Dynamic<tinte,tker,tvis,true>(); else InteractionPeri_Dynamic<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Dynamic<tinte,tker,tvis,true>(); else InteractionPeri_Dynamic<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Dynamic<tinte,tker,tvis,true>(); else InteractionPeri_Dynamic<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Dynamic<tinte,tker,tvis,true>(); else InteractionPeri_Dynamic<tinte,tker,tvis,false>();
        }
      }
    }else if(tinter==INTER_ForcesCorr){ const TpInter tinte=INTER_ForcesCorr;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Dynamic<tinte,tker,tvis,true>(); else InteractionPeri_Dynamic<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Dynamic<tinte,tker,tvis,true>(); else InteractionPeri_Dynamic<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Dynamic<tinte,tker,tvis,true>(); else InteractionPeri_Dynamic<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Dynamic<tinte,tker,tvis,true>(); else InteractionPeri_Dynamic<tinte,tker,tvis,false>();
        }
      }
    }else if(tinter==INTER_Shepard){ const TpInter tinte=INTER_Shepard;
      if(TKernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Dynamic<tinte,tker,tvis,true>(); else InteractionPeri_Dynamic<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Dynamic<tinte,tker,tvis,true>(); else InteractionPeri_Dynamic<tinte,tker,tvis,false>();
        }
      }else if(TKernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
        if(TVisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
          if(WithFloating)InteractionPeri_Dynamic<tinte,tker,tvis,true>(); else InteractionPeri_Dynamic<tinte,tker,tvis,false>();
        }else if(TVisco==VISCO_LaminarSPS){ const TpVisco tvis=VISCO_LaminarSPS; 
          if(WithFloating)InteractionPeri_Dynamic<tinte,tker,tvis,true>(); else InteractionPeri_Dynamic<tinte,tker,tvis,false>();
        }
      }
    }
  }
}

//==============================================================================
/// Interactions with periodic zone Single-Core.
//==============================================================================
template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void JSphCpu::InteractionPeri_Single(){
  const int np=PeriodicZone->GetListNp();
  const int npb=PeriodicZone->GetListNpb();
  const unsigned *list=PeriodicZone->GetList();
  const int boundini=int(PeriodicZone->GetListBoundIni());
  const int fluidini=int(PeriodicZone->GetListFluidIni())-npb;
  const unsigned ncx=PeriodicZone->GetNc1();
  const unsigned *cellpart=PeriodicZone->GetListCellPart();

  for(int p=0;p<np;p++){
    const byte kind2ini=(p<npb? 2: 1);
    const unsigned i=(list? list[p]: p+(p<npb? boundini: fluidini));  
    const TpParticle tpi=GetTpParticle(i);
    const unsigned cel=cellpart[p];
    //-Gets limits of interaction. 
    const unsigned cx=(cel>>16);
    const unsigned cy=(cel&0xffff);
    const unsigned cxini=cx-Hdiv;
    const unsigned cxfin=cx+Hdiv+1;
    const unsigned yini=cy-Hdiv;
    const unsigned yfin=cy+Hdiv+1;
    float viscdt=0;
    for(unsigned y=yini;y<yfin;y++){
      const unsigned ymod=ncx*y;
      float viscdtcf=InteractCelijPeri<tinter,tker,tvis,floating>(i,tpi,cxini+ymod,cxfin+ymod,kind2ini);
      viscdt=max(viscdt,viscdtcf);
    }
    //-Accumulates value of viscdt from force computation.
    if(tinter==INTER_Forces||tinter==INTER_ForcesCorr)ViscDtThread[0]=max(ViscDtThread[0],viscdt);
  }
}

#ifdef _WITHOMP
//==============================================================================
/// Interactions with periodic zone OpenMP Static.
//==============================================================================
template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void JSphCpu::InteractionPeri_Static(){
  const int np=PeriodicZone->GetListNp();
  const int npb=PeriodicZone->GetListNpb();
  const unsigned *list=PeriodicZone->GetList();
  const int boundini=int(PeriodicZone->GetListBoundIni());
  const int fluidini=int(PeriodicZone->GetListFluidIni())-npb;
  const unsigned ncx=PeriodicZone->GetNc1();
  const unsigned *cellpart=PeriodicZone->GetListCellPart();

  #pragma omp parallel for schedule (static,50)
  for(int p=0;p<np;p++){
    const byte kind2ini=(p<npb? 2: 1);
    const unsigned i=(list? list[p]: p+(p<npb? boundini: fluidini));  
    const TpParticle tpi=GetTpParticle(i);
    const unsigned cel=cellpart[p];
    //-Gets limits of interaction.
    const unsigned cx=(cel>>16);
    const unsigned cy=(cel&0xffff);
    const unsigned cxini=cx-Hdiv;
    const unsigned cxfin=cx+Hdiv+1;
    const unsigned yini=cy-Hdiv;
    const unsigned yfin=cy+Hdiv+1;
    float viscdt=0;
    for(unsigned y=yini;y<yfin;y++){
      const unsigned ymod=ncx*y;
      float viscdtcf=InteractCelijPeri<tinter,tker,tvis,floating>(i,tpi,cxini+ymod,cxfin+ymod,kind2ini);
      viscdt=max(viscdt,viscdtcf);
    }
    //-Accumulates value of viscdt from force computation.
    if(tinter==INTER_Forces||tinter==INTER_ForcesCorr){
      int cth=omp_get_thread_num()*STRIDE_OMP;
      ViscDtThread[cth]=max(ViscDtThread[cth],viscdt);
    }
  }
}

//==============================================================================
/// Interactions with periodic zone OpenMP Dynamic.
//==============================================================================
template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void JSphCpu::InteractionPeri_Dynamic(){
  const int np=PeriodicZone->GetListNp();
  const int npb=PeriodicZone->GetListNpb();
  const unsigned *list=PeriodicZone->GetList();
  const int boundini=int(PeriodicZone->GetListBoundIni());
  const int fluidini=int(PeriodicZone->GetListFluidIni())-npb;
  const unsigned ncx=PeriodicZone->GetNc1();
  const unsigned *cellpart=PeriodicZone->GetListCellPart();

  #pragma omp parallel for schedule (dynamic,50)
  for(int p=0;p<np;p++){
    const byte kind2ini=(p<npb? 2: 1);
    const unsigned i=(list? list[p]: p+(p<npb? boundini: fluidini));  
    const TpParticle tpi=GetTpParticle(i);
    const unsigned cel=cellpart[p];
    //-Gets limits of interaction.
    const unsigned cx=(cel>>16);
    const unsigned cy=(cel&0xffff);
    const unsigned cxini=cx-Hdiv;
    const unsigned cxfin=cx+Hdiv+1;
    const unsigned yini=cy-Hdiv;
    const unsigned yfin=cy+Hdiv+1;
    float viscdt=0;
    for(unsigned y=yini;y<yfin;y++){
      const unsigned ymod=ncx*y;
      float viscdtcf=InteractCelijPeri<tinter,tker,tvis,floating>(i,tpi,cxini+ymod,cxfin+ymod,kind2ini);
      viscdt=max(viscdt,viscdtcf);
    }
    //-Accumulates value of viscdt from force computation.
    if(tinter==INTER_Forces||tinter==INTER_ForcesCorr){
      int cth=omp_get_thread_num()*STRIDE_OMP;
      ViscDtThread[cth]=max(ViscDtThread[cth],viscdt);
    }
  }
}
#endif

//==============================================================================
/// Interactions with periodic zone Single-Core (one particle with other cells).
//==============================================================================
template<TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> float JSphCpu::InteractCelijPeri(unsigned i,TpParticle tpi,unsigned boxini,unsigned boxfin,byte kind2ini){
  float viscdt=0;
  for(byte kind=kind2ini;kind<=2;kind++){
    int jbegin=int(PeriodicZone->CellBegin(boxini,kind));
    int jend=int(PeriodicZone->CellBegin(boxfin,kind));
    if(jbegin<jend){
      if(tinter==INTER_Shepard){  //-Shepard only fluid-fluid (discards floating).
        if(tpi==PART_Fluid)for(int j=jbegin;j<jend;j++){
          float drx=Posi[i].x-Posj[j].x, dry=Posi[i].y-Posj[j].y, drz=Posi[i].z-Posj[j].z;
          float rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=Fourh2&&rr2>=1e-18f&&GetTpParticleCode(Codej[j])==PART_Fluid)ComputeForcesShepard<false,tker>(i,j,drx,dry,drz,rr2,0);
        }
      }
      else{
        for(int j=jbegin;j<jend;j++){
          float drx=Posi[i].x-Posj[j].x, dry=Posi[i].y-Posj[j].y, drz=Posi[i].z-Posj[j].z;
          float rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=Fourh2&&rr2>=1e-18f){
            float viscdtcf=ComputeForces<false,tinter,tker,tvis,floating>(tpi,GetTpParticleCode(Codej[j]),i,j,drx,dry,drz,rr2,0,0);
            viscdt=max(viscdt,viscdtcf);
          }
        }
      }
    }
  }
  return(viscdt);
}

//==============================================================================
/// Interactions with periodic zone Single-Core (one particle with its own cell).
//==============================================================================
template<bool tsym,TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void JSphCpu::InteractSelf(int box,byte kind,byte kind2ini){
  float viscdt=0; 
#ifdef _WITHOMP
  const int cth=omp_get_thread_num(),offset=(OmpMode!=OMPM_Single? Np*cth: 0),offsetf=(OmpMode!=OMPM_Single? (Np-Npb)*cth: 0),offsetsh=(tinter!=INTER_Shepard? 0: offsetf);
#else
  const int cth=0,offset=0,offsetf=0,offsetsh=0;
#endif
  int ibegin=int(CellDiv->CellBegin(box,kind));
  int iend=int(CellDiv->CellBegin(box+1,kind));
  for(byte kind2=max(kind,kind2ini);kind2<=2;kind2++){
    int jbegin,jend;
    if(kind!=kind2){
      jbegin=int(CellDiv->CellBegin(box,kind2));
      jend=int(CellDiv->CellBegin(box+1,kind2));
    }
    else jend=iend;
    for(int i=ibegin;i<iend;i++){
      TpParticle tpi=GetTpParticle(i);
      if(tsym&&kind==kind2)jbegin=i+1;
      if(tinter==INTER_Shepard){  //-Shepard only fluid-fluid (discards floating).
        if(tpi==PART_Fluid)for(int j=jbegin;j<jend;j++){
          float drx=Pos[i].x-Pos[j].x, dry=Pos[i].y-Pos[j].y, drz=Pos[i].z-Pos[j].z;
          float rr2=drx*drx+dry*dry+drz*drz;
          //if(i!=j)Nite++; //>dbg
          if(rr2<=Fourh2&&rr2>=1e-18f&&GetTpParticle(j)==PART_Fluid)ComputeForcesShepard<tsym,tker>(i,j,drx,dry,drz,rr2,offsetsh);
        }
      }
      else{
        for(int j=jbegin;j<jend;j++){
          float drx=Pos[i].x-Pos[j].x, dry=Pos[i].y-Pos[j].y, drz=Pos[i].z-Pos[j].z;
          float rr2=drx*drx+dry*dry+drz*drz;
          //if(i!=j)Nite++; //>dbg
          if(rr2<=Fourh2&&rr2>=1e-18f){
            float viscdtcf=ComputeForces<tsym,tinter,tker,tvis,floating>(tpi,GetTpParticle(j),i,j,drx,dry,drz,rr2,offset,offsetf);
            viscdt=max(viscdt,viscdtcf);
          }
        }
      }
    }
  }
  if(tinter==INTER_Forces||tinter==INTER_ForcesCorr)ViscDtThread[cth*STRIDE_OMP]=max(ViscDtThread[cth*STRIDE_OMP],viscdt);
}

//==============================================================================
/// Interactions of one cell with other cells.
//==============================================================================
template<bool tsym,TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> void JSphCpu::InteractCelij(int ibegin,int iend,int box1,int box2,byte kind2ini){
  float viscdt=0;
#ifdef _WITHOMP
  const int cth=omp_get_thread_num(),offset=(OmpMode!=OMPM_Single? Np*cth: 0),offsetf=(OmpMode!=OMPM_Single? (Np-Npb)*cth: 0),offsetsh=(tinter!=INTER_Shepard? 0: offsetf);
#else
  const int cth=0,offset=0,offsetf=0,offsetsh=0;
#endif
  for(byte kind=kind2ini;kind<=2;kind++){
    int jbegin=int(CellDiv->CellBegin(box1,kind));
    int jend=int(CellDiv->CellBegin(box2+1,kind));
    if(jbegin<jend){
      for(int i=ibegin;i<iend;i++){
        TpParticle tpi=GetTpParticle(i);
        if(tinter==INTER_Shepard){//-Shepard only fluid-fluid (discards floating).
          if(tpi==PART_Fluid)for(int j=jbegin;j<jend;j++){
            float drx=Pos[i].x-Pos[j].x, dry=Pos[i].y-Pos[j].y, drz=Pos[i].z-Pos[j].z;
            float rr2=drx*drx+dry*dry+drz*drz;
            //if(i!=j)Nite++; //>dbg
            if(rr2<=Fourh2&&rr2>=1e-18f&&GetTpParticle(j)==PART_Fluid)ComputeForcesShepard<tsym,tker>(i,j,drx,dry,drz,rr2,offsetsh);
          }
        }
        else{
          for(int j=jbegin;j<jend;j++){
            float drx=Pos[i].x-Pos[j].x, dry=Pos[i].y-Pos[j].y, drz=Pos[i].z-Pos[j].z;
            float rr2=drx*drx+dry*dry+drz*drz;
            //if(i!=j)Nite++; //>dbg
            if(rr2<=Fourh2&&rr2>=1e-18f){
              float viscdtcf=ComputeForces<tsym,tinter,tker,tvis,floating>(tpi,GetTpParticle(j),i,j,drx,dry,drz,rr2,offset,offsetf);
              viscdt=max(viscdt,viscdtcf);
            }
          }
        }
      }
    }
  }
  if(tinter==INTER_Forces||tinter==INTER_ForcesCorr)ViscDtThread[cth*STRIDE_OMP]=max(ViscDtThread[cth*STRIDE_OMP],viscdt);
}

//==============================================================================
/// Computes kernel and density summation for Shepard filter.
//==============================================================================
template<bool tsym,TpKernel tker> void JSphCpu::ComputeForcesShepard(int i,int j,float drx,float dry,float drz,float rr2,int offsetsh){
  float *fdwab=FdWab+offsetsh;
  float *fdrhop=FdRhop+offsetsh;
  const float rad=sqrt(rr2);
  const float qq=rad/H;
  float wab;
  if(tker==KERNEL_Cubic){    //-Cubic kernel.
    if(rad>H){
      float wqq1=2.0f-qq;
      wab=CubicCte.a24*(wqq1*wqq1*wqq1);
    }
    else wab=CubicCte.a2*(1.0f+(0.75f*qq-1.5f)*qq*qq);//float wqq2=qq*qq;  wab=CubicCte.a2*(1.0f-1.5f*wqq2+0.75f*(wqq2*qq));   
  }
  if(tker==KERNEL_Wendland){ //-Wendland kernel.
    float wqq=2*qq+1;
    float wqq1=1.f-0.5f*qq;
    float wqq2=wqq1*wqq1;
    wab=WendlandCte.awen*wqq*wqq2*wqq2;
  }
  const int i2=i-Npb,j2=j-Npb;
  fdwab[i2]+=wab*(MassFluid/Rhopj[j]); 
  fdrhop[i2]+=wab;           //-MassFluid is multiplied in the end FdRhop[i]+=wab*MassFluid.
  if(tsym){
    fdwab[j2]+=wab*(MassFluid/Rhopi[i]);
    fdrhop[j2]+=wab;         //-MassFluid is multiplied in the end  FdRhop[j]+=wab*MassFluid.
  }
}

//==============================================================================
/// Computes acceleration and density derivative during particle interactions.
/// [\ref INTER_Forces,\ref INTER_ForcesCorr]
//==============================================================================
template<bool tsym,TpInter tinter,TpKernel tker,TpVisco tvis,bool floating> float JSphCpu::ComputeForces(TpParticle tpi,TpParticle tpj,int i,int j,float drx,float dry,float drz,float rr2,int offset,int offsetf){
  float viscdt=0;
  float *ar=Ar+offset;
  tfloat3 *ace=Ace+offset;
  tfloat3 *velxcor=VelXcor+offset;
  float *delta=(Delta? Delta+offset: NULL);

  if(tinter==INTER_Forces||tinter==INTER_ForcesCorr){
    const bool computei=(tpi&PART_BoundFt_Fluid)!=0;            //-Particle i is Fluid or BoundFt. 
    const bool computej=tsym&&((tpj&PART_BoundFt_Fluid)!=0);    //-Particle j is Fluid or BoundFt.   
    float prs=PrRhopi[i]+PrRhopj[j];
    float wab,frx,fry,frz;    
    {//===== Kernel =====
      float rad=sqrt(rr2);
      float qq=rad/H;
      float fac;
      if(tker==KERNEL_Cubic){//-Cubic kernel.
        if(rad>H){
          float wqq1=2.0f-qq;
          float wqq2=wqq1*wqq1;
          wab=CubicCte.a24*(wqq2*wqq1);
          fac=CubicCte.c2*wqq2/rad;
        }
        else{
          float wqq2=qq*qq;
          float wqq3=wqq2*qq;
          wab=CubicCte.a2*(1.0f-1.5f*wqq2+0.75f*wqq3);
          fac=(CubicCte.c1*qq+CubicCte.d1*wqq2)/rad;
        }
        //-Tensile correction.
        float fab=wab*CubicCte.od_wdeltap;
        fab*=fab; fab*=fab; //fab=fab^4
        prs+=fab*(Tensili[i]+Tensilj[j]);
      }
      if(tker==KERNEL_Wendland){//-Wendland kernel.
        float wqq=2.f*qq+1.f;
        float wqq1=1.f-0.5f*qq;
        float wqq2=wqq1*wqq1;
        wab=WendlandCte.awen*wqq*wqq2*wqq2;
        fac=WendlandCte.bwen*qq*wqq2*wqq1/rad;
      }
      frx=fac*drx; fry=fac*dry; frz=fac*drz;
    }
    //===== Mass of particles ===== 
    float massi=(tpi==PART_Fluid? MassFluid: MassBound);
    float massj=(tpj==PART_Fluid? MassFluid: MassBound);
    float massfti=1,massftj=1;
    if(floating){
      if(tpi==PART_BoundFt)massfti=massi=FtObjs[CODE_GetTypeValue(Codei[i])].massp;
      if(tpj==PART_BoundFt)massftj=massj=FtObjs[CODE_GetTypeValue(Codej[j])].massp;
    }
    {//===== Aceleration ===== 
      if(computei){          //-Particle i is Fluid.
        const float p_vpmi=-prs*massj*massfti;
        ace[i].x+=p_vpmi*frx; ace[i].y+=p_vpmi*fry; ace[i].z+=p_vpmi*frz;
      }
      if(computej){                          
        const float p_vpmj=prs*massi*massftj;
        ace[j].x+=p_vpmj*frx; ace[j].y+=p_vpmj*fry; ace[j].z+=p_vpmj*frz;
      }
    }
    //===== Density derivative =====
    const float dvx=Veli[i].x-Velj[j].x, dvy=Veli[i].y-Velj[j].y, dvz=Veli[i].z-Velj[j].z;
    const float dot2=(dvx*frx+dvy*fry+dvz*frz);
    ar[i]+=massj*dot2;
    if(tsym)ar[j]+=massi*dot2;

    const float cbar=(Csoundi[i]+Csoundj[j])*0.5f;
    //===== DeltaSPH =====
    if(delta){
      const float dot3=(drx*frx+dry*fry+drz*frz);
      const float deltaij=Delta2H*cbar/(rr2+Eta2)*dot3;
      const float rhopioverj=(Rhopi[i]/Rhopj[j]);
      const bool fluidi=((tpi&PART_Fluid)!=0),fluidj=((tpj&PART_Fluid)!=0);      
      if(fluidi&&fluidj){
        delta[i]+=deltaij*(rhopioverj-1)*massj;
        if(tsym)delta[j]+=deltaij*(1/rhopioverj-1)*massi; 
      }
      if(!fluidi||!fluidj){
        delta[i]=FLT_MAX;
        if(tsym)delta[j]=FLT_MAX;
      }
    }

    const float robar=(Rhopi[i]+Rhopj[j])*0.5f;
    {//===== Viscosity ===== 
      const float dot=drx*dvx + dry*dvy + drz*dvz;
      //-Artificial viscosity .
      if(tvis==VISCO_Artificial && dot<0){
        float amubar=H*dot/(rr2+Eta2);
        float pi_visc=-Visco*cbar*amubar/robar;
        if(computei){
          const float v=-massj*pi_visc*massfti; ace[i].x+=v*frx; ace[i].y+=v*fry; ace[i].z+=v*frz;
        }
        if(computej){ 
          const float v=massi*pi_visc*massftj;  ace[j].x+=v*frx; ace[j].y+=v*fry; ace[j].z+=v*frz; 
        }
      }
      //-Laminar+SPS viscosity.   
      if(tvis==VISCO_LaminarSPS){
         tsymatrix3f *csph=Csph+offsetf;
         const float temp=2.0f*Visco/((rr2+Eta2)*robar);
         if(computei){
           const float vtemp=massj*temp*(drx*frx+dry*fry+drz*frz);  
           ace[i].x+=vtemp*dvx; ace[i].y+=vtemp*dvy; ace[i].z+=vtemp*dvz;
         }
         if(computej){ 
           const float vtemp=-massi*temp*(drx*frx+dry*fry+drz*frz); 
           ace[j].x+=vtemp*dvx; ace[j].y+=vtemp*dvy; ace[j].z+=vtemp*dvz; 
         }
         //-SPS turbulence model.  
         float tau_xx=0,tau_xy=0,tau_xz=0,tau_yy=0,tau_yz=0,tau_zz=0;
         if(tpi==PART_Fluid){ 
           tau_xx+=Taui[i].xx; tau_xy+=Taui[i].xy; tau_xz+=Taui[i].xz;
           tau_yy+=Taui[i].yy; tau_yz+=Taui[i].yz; tau_zz+=Taui[i].zz;
         }
         if(tpj==PART_Fluid){ 
           tau_xx+=Tauj[j].xx; tau_xy+=Tauj[j].xy; tau_xz+=Tauj[j].xz;
           tau_yy+=Tauj[j].yy; tau_yz+=Tauj[j].yz; tau_zz+=Tauj[j].zz;
         }
         if(computei){  
           ace[i].x+=massj*massfti*(tau_xx*frx+tau_xy*fry+tau_xz*frz);
           ace[i].y+=massj*massfti*(tau_xy*frx+tau_yy*fry+tau_yz*frz);
           ace[i].z+=massj*massfti*(tau_xz*frx+tau_yz*fry+tau_zz*frz);
         }
         if(computej){ 
           ace[j].x+=-massi*massftj*(tau_xx*frx+tau_xy*fry+tau_xz*frz);
           ace[j].y+=-massi*massftj*(tau_xy*frx+tau_yy*fry+tau_yz*frz);
           ace[j].z+=-massi*massftj*(tau_xz*frx+tau_yz*fry+tau_zz*frz);
         }
         //-Velocity gradients.       
         if(computei){
           const int i2=i-Npb;
           const float volj=-massj/Rhop[j];
           float dv=dvx*volj; csph[i2].xx+=dv*frx; csph[i2].xy+=dv*fry; csph[i2].xz+=dv*frz;
                 dv=dvy*volj; csph[i2].xy+=dv*frx; csph[i2].yy+=dv*fry; csph[i2].yz+=dv*frz;
                 dv=dvz*volj; csph[i2].xz+=dv*frx; csph[i2].yz+=dv*fry; csph[i2].zz+=dv*frz;
           // to compute tau terms we assume that csph.xy=csph.dudy+csph.dvdx, csph.xz=csph.dudz+csph.dwdx, csph.yz=csph.dvdz+csph.dwdy
           // so only 6 elements are needed instead of 3x3.
         }
         if(computej){ 
           const int j2=j-Npb;
           const float voli=-massi/Rhop[i];
           float dv=dvx*voli; csph[j2].xx+=dv*frx; csph[j2].xy+=dv*fry; csph[j2].xz+=dv*frz;
                 dv=dvy*voli; csph[j2].xy+=dv*frx; csph[j2].yy+=dv*fry; csph[j2].yz+=dv*frz;
                 dv=dvz*voli; csph[j2].xz+=dv*frx; csph[j2].yz+=dv*fry; csph[j2].zz+=dv*frz;
         }
       }
       viscdt=dot/(rr2+Eta2);
    }

    //===== XSPH correction =====
    if(tinter==INTER_Forces){//-XSPH correction only for Verlet or Symplectic-Predictor.
      const float wab_rhobar=wab/robar;
      if(computei){
        float wab_rhobar_mass=massj*wab_rhobar;
        velxcor[i].x-=wab_rhobar_mass * dvx;
        velxcor[i].y-=wab_rhobar_mass * dvy;
        velxcor[i].z-=wab_rhobar_mass * dvz;
      }
      if(computej){
        float wab_rhobar_mass=massi*wab_rhobar;
        velxcor[j].x+=wab_rhobar_mass * dvx;
        velxcor[j].y+=wab_rhobar_mass * dvy;
        velxcor[j].z+=wab_rhobar_mass * dvz;
      }
    }
  }
  return(viscdt);
}

//==============================================================================
/// Computes new values of position and velocity using VERLET algorithm.
//==============================================================================
void JSphCpu::ComputeVerletVars(const tfloat3 *vel1,const tfloat3 *vel2,float dt,float dt2,tfloat3 *velnew){
  const float dtsq_05=0.5f*dt*dt;
  const int np=int(Np);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (static)
  #endif
  for(int p=int(Npb);p<np;p++){
    if(!WithFloating||Idp[p]>=CaseNbound){  //-PART_Fluid.
      float dx=(vel1[p].x+VelXcor[p].x*Eps) * dt + Ace[p].x*dtsq_05;
      float dy=(vel1[p].y+VelXcor[p].y*Eps) * dt + Ace[p].y*dtsq_05;
      float dz=(vel1[p].z+VelXcor[p].z*Eps) * dt + Ace[p].z*dtsq_05;
      if(abs(dx)>MovLimit||abs(dy)>MovLimit||abs(dz)>MovLimit)Code[p]=CODE_SetOutMove(Code[p]);
      Pos[p].x+=dx; Pos[p].y+=dy; Pos[p].z+=dz;
      velnew[p].x=vel2[p].x+Ace[p].x*dt2;
      velnew[p].y=vel2[p].y+Ace[p].y*dt2;
      velnew[p].z=vel2[p].z+Ace[p].z*dt2;
    }
    else velnew[p]=vel1[p];                 //-PART_BoundFt.
  }
}

//==============================================================================
/// Updates particles according to forces and dt using VERLET. 
//==============================================================================
void JSphCpu::ComputeVerlet(bool rhopbound,float dt){
  TmcStart(Timers,TMC_SuComputeStep);
  VerletStep++;
  if(VerletStep<VerletSteps){
    float twodt=dt+dt;
    ComputeVerletVars(Vel,VelM1,dt,twodt,VelM1);
    ComputeRhop(RhopM1,RhopM1,twodt,rhopbound);
  }
  else{
    ComputeVerletVars(Vel,Vel,dt,dt,VelM1);
    ComputeRhop(RhopM1,Rhop,dt,rhopbound);
    VerletStep=0;
  }
  //-New values are computed in VelM1 & RhopM1.
  swap(Vel,VelM1);           //Swaps Vel <= VelM1.
  swap(Rhop,RhopM1);         //Swaps Rhop <= RhopM1.
  if(CaseNmoving)memset(Vel,0,sizeof(tfloat3)*Npb);   //Velg[]=0 for boundaries.
  TmcStop(Timers,TMC_SuComputeStep);
}

//==============================================================================
/// Updates particles according to forces and dt using SYMPLECTIC (Predictor step).
//==============================================================================
void JSphCpu::ComputeSymplecticPre(bool rhopbound,float dt){
  TmcStart(Timers,TMC_SuComputeStep);
  //-Changes data to variables Pre to compute new data.
  swap(PosPre,Pos);          //Swaps PosPre[] <= Pos[].
  swap(VelPre,Vel);          //Swaps VelPre[] <= Vel[].
  swap(RhopPre,Rhop);        //Swaps RhopPre[]<= Rhop[].
  //-Computes new data of particles.
  const float dt05=dt*.5f;
  ComputeRhop(Rhop,RhopPre,dt05,rhopbound);
  const int np=int(Np);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (static)
  #endif
  for(int p=int(Npb);p<np;p++){
    if(!WithFloating||Idp[p]>=CaseNbound){  //-PART_Fluid.
      Vel[p].x=VelPre[p].x + Ace[p].x * dt05; 
      Vel[p].y=VelPre[p].y + Ace[p].y * dt05; 
      Vel[p].z=VelPre[p].z + Ace[p].z * dt05; 
      float dx=(VelPre[p].x+VelXcor[p].x*Eps) * dt05; 
      float dy=(VelPre[p].y+VelXcor[p].y*Eps) * dt05; 
      float dz=(VelPre[p].z+VelXcor[p].z*Eps) * dt05; 
      if(abs(dx)>MovLimit||abs(dy)>MovLimit||abs(dz)>MovLimit)Code[p]=CODE_SetOutMove(Code[p]);
      Pos[p].x=PosPre[p].x + dx; 
      Pos[p].y=PosPre[p].y + dy; 
      Pos[p].z=PosPre[p].z + dz; 
    }
    else{                                   //-PART_BoundFt.
      Vel[p]=VelPre[p];
      Pos[p]=PosPre[p];
    }
  }
  memcpy(Pos,PosPre,Npb*sizeof(tfloat3));
  memcpy(Vel,VelPre,Npb*sizeof(tfloat3));
  TmcStop(Timers,TMC_SuComputeStep);
}
//==============================================================================
/// Updates particles according to forces and dt using SYMPLECTIC (Corrector step).
//==============================================================================
void JSphCpu::ComputeSymplecticCorr(bool rhopbound,float dt){
  TmcStart(Timers,TMC_SuComputeStep);
  const float dt05=dt*.5f;
  ComputeRhopEpsilon(Rhop,RhopPre,dt,rhopbound);
  const int np=int(Np);
  #ifdef _WITHOMP
    #pragma omp parallel for schedule (static)
  #endif
  for(int p=Npb;p<np;p++){
    if(!WithFloating||Idp[p]>=CaseNbound){  //-PART_Fluid.
      Vel[p].x=VelPre[p].x + Ace[p].x * dt; 
      Vel[p].y=VelPre[p].y + Ace[p].y * dt; 
      Vel[p].z=VelPre[p].z + Ace[p].z * dt; 
      float dx=(VelPre[p].x+Vel[p].x) * dt05; 
      float dy=(VelPre[p].y+Vel[p].y) * dt05; 
      float dz=(VelPre[p].z+Vel[p].z) * dt05; 
      if(abs(dx)>MovLimit||abs(dy)>MovLimit||abs(dz)>MovLimit)Code[p]=CODE_SetOutMove(Code[p]);
      Pos[p].x=PosPre[p].x + dx; 
      Pos[p].y=PosPre[p].y + dy; 
      Pos[p].z=PosPre[p].z + dz; 
    }
    else{                                   //-PART_BoundFt.
      Vel[p]=VelPre[p];
      Pos[p]=PosPre[p];
    }
  }
  if(CaseNmoving)memset(Vel,0,Npb*sizeof(tfloat3));
  TmcStop(Timers,TMC_SuComputeStep);
}

//==============================================================================
/// Updates new value of density.
//==============================================================================
void JSphCpu::ComputeRhop(float* rhopnew,const float* rhopold,float armul,bool rhopbound){
  for(unsigned p=Npb;p<Np;p++)rhopnew[p]=rhopold[p]+Ar[p]*armul; 
  if(rhopbound)for(unsigned p=0;p<Npb;p++){
    rhopnew[p]=rhopold[p]+Ar[p]*armul;
    if(rhopnew[p]<Rhop0)rhopnew[p]=Rhop0;   //-To prevent absorption of fluid particles by boundaries.
  }
  else if(rhopnew!=rhopold)memcpy(rhopnew,rhopold,sizeof(float)*Npb);
  if(WithFloating){                         //-To prevent absorption of fluid particles by floating.
    CalcFtRidp();
    for(unsigned c=0;c<CaseNfloat;c++){ 
      const unsigned p=FtRidp[c]; 
      if(p!=UINT_MAX&&rhopnew[p]<Rhop0)rhopnew[p]=Rhop0; 
    }
  }
}

//==============================================================================
/// Updates new value of density using epsilon (for Corrector step of Symplectic).
//==============================================================================
void JSphCpu::ComputeRhopEpsilon(float* rhopnew,const float* rhopold,float armul,bool rhopbound){
  for(unsigned p=Npb;p<Np;p++){
    const float epsilon_rdot=(-Ar[p]/rhopnew[p])*armul;
    rhopnew[p]=rhopold[p] * (2.f-epsilon_rdot)/(2.f+epsilon_rdot);
  }
  if(rhopbound)for(unsigned p=0;p<Npb;p++){
    const float epsilon_rdot=(-Ar[p]/rhopnew[p])*armul;
    rhopnew[p]=rhopold[p] * (2.f-epsilon_rdot)/(2.f+epsilon_rdot);
    if(rhopnew[p]<Rhop0)rhopnew[p]=Rhop0;   //-To prevent absorption of fluid particles by boundaries.
  }
  else memcpy(rhopnew,rhopold,sizeof(float)*Npb);
  if(WithFloating){                         //-To prevent absorption of fluid particles by floating.
    CalcFtRidp();
    for(unsigned c=0;c<CaseNfloat;c++){ 
      const unsigned p=FtRidp[c]; 
      if(p!=UINT_MAX&&rhopnew[p]<Rhop0)rhopnew[p]=Rhop0; 
    }
  }
}

//==============================================================================
/// Computes sub-particle stress tensor (Tau) for SPS turbulence model.   
//==============================================================================
void JSphCpu::SPSCalcTau(){       
  const int npf=(Np-Npb);
#ifdef USE_OPENMP
  #pragma omp parallel for schedule (static)
#endif
  for(int p=0;p<npf;p++){
    const tsymatrix3f csph=Csph[p];
    const float pow1=csph.xx*csph.xx + csph.yy*csph.yy + csph.zz*csph.zz;
    const float prr=pow1+pow1 + csph.xy*csph.xy + csph.xz*csph.xz + csph.yz*csph.yz;
    const float visc_SPS=SpsSmag*sqrt(prr);
    const float div_u=csph.xx+csph.yy+csph.zz;
    const float sps_k=(2.0f/3.0f)*visc_SPS*div_u;
    const float sps_Blin=SpsBlin*prr;
    const float sumsps=-(sps_k+sps_Blin);
    const float twovisc_SPS=(visc_SPS+visc_SPS);
    const float one_rho2 = 1.0f/Rhop[p+Npb];   
    const unsigned p2=p+Npb;
    Tau[p2].xx=one_rho2*(twovisc_SPS*csph.xx +sumsps);
    Tau[p2].xy=one_rho2*(visc_SPS*csph.xy);
    Tau[p2].xz=one_rho2*(visc_SPS*csph.xz);
    Tau[p2].yy=one_rho2*(twovisc_SPS*csph.yy +sumsps);
    Tau[p2].yz=one_rho2*(visc_SPS*csph.yz);
    Tau[p2].zz=one_rho2*(twovisc_SPS*csph.zz +sumsps);
  }
}

//==============================================================================
/// Updates new value of density using Shepard (for a range of particles).
//==============================================================================
void JSphCpu::ComputeShepard(int pini,int pfin){
  //const int n=pfin-pini;
  //const int npf=Np-Npb;
  //-Adjust value of Rhop[] in fluids.
  if(!WithFloating){
    #pragma omp parallel for schedule (static)
    for(int p=pini;p<pfin;p++){
      //FdWab[p]+=CteShepard*FdVol[p];
      //FdRhop[p]+=CteShepard;
      //Rhop[p+Npb]=(FdRhop[p]*MassFluid)/FdWab[p];
      Rhop[p]=((FdRhop[p-Npb]+CteShepard)*MassFluid)/(FdWab[p-Npb]+(CteShepard* (MassFluid/Rhop[p]) ));
    }
  }
  else{//-Filters floating to use only fluids.
    #pragma omp parallel for schedule (static)
    for(int p=pini;p<pfin;p++){
      if(Idp[p]>=CaseNbound){
        //FdWab[p]+=CteShepard*FdVol[p];
        //FdRhop[p]+=CteShepard;
        //Rhop[p+Npb]=(FdRhop[p]*MassFluid)/FdWab[p];
        Rhop[p]=((FdRhop[p-Npb]+CteShepard)*MassFluid)/(FdWab[p-Npb]+(CteShepard* (MassFluid/Rhop[p]) ));
      }
    }
  }
}

//==============================================================================
/// Computes a variable DT.
//==============================================================================
float JSphCpu::DtVariable(){
  float dt=(CsoundMax||ViscDtMax? CFLnumber*(H/(CsoundMax+H*ViscDtMax)): FLT_MAX);
  if(DtFixed)dt=DtFixed->GetDt(TimeStep,dt);
  //char cad[512]; sprintf(cad,"____Dt[%u]:%f  csound:%f  viscdt:%f",Nstep,dt,CsoundMax,ViscDt); Log->Print(cad);
  if(dt<DtMin){ dt=DtMin; DtModif++; }
  return(dt);
}

//==============================================================================
/// Processes movement of moving boundary particles.
//==============================================================================
void JSphCpu::RunMotion(float stepdt){
  TmcStart(Timers,TMC_SuMotion);
  if(Motion->ProcesTime(TimeStep+MotionTimeMod,stepdt)){
    unsigned nmove=Motion->GetMovCount();
    //{ char cad[256]; sprintf(cad,"----RunMotion[%u]>  nmove:%u",Nstep,nmove); Log->Print(cad); }
    if(nmove){
      CalcRidpMoving();
      //-Movevement of boundary particles.
      for(unsigned c=0;c<nmove;c++){
        unsigned ref;
        tfloat3 mvsimple;
        tmatrix4f mvmatrix;
        if(Motion->GetMov(c,ref,mvsimple,mvmatrix)){  //-Simple movement.
          mvsimple=OrderCode(mvsimple);
          const unsigned pini=MotionObjBegin[ref]-CaseNfixed;
          const unsigned pfin=MotionObjBegin[ref+1]-CaseNfixed;
          //printf("*******mov>%d -> %d\n",MotionObjBegin[ref],plast);
          if(Simulate2D)mvsimple.y=0;
          tfloat3 mvvel=mvsimple/TFloat3(stepdt);
          bool out=(abs(mvsimple.x)>MovLimit||abs(mvsimple.y)>MovLimit||abs(mvsimple.z)>MovLimit);
          for(unsigned mp=pini;mp<pfin;mp++){
            unsigned pid=RidpMoving[mp];  //printf("id:%d -> pid:%d\n",id,pid);
            if(pid!=UINT_MAX){
              if(out)Code[pid]=CODE_SetOutMove(Code[pid]);
              Pos[pid].x+=mvsimple.x; Pos[pid].y+=mvsimple.y; Pos[pid].z+=mvsimple.z;
              Vel[pid]=mvvel;
            }
          }
        }
        else{                                         //-Movement with matrix.
          mvmatrix=OrderCode(mvmatrix);
          const unsigned pini=MotionObjBegin[ref]-CaseNfixed;
          const unsigned pfin=MotionObjBegin[ref+1]-CaseNfixed;
          for(unsigned mp=pini;mp<pfin;mp++){
            unsigned pid=RidpMoving[mp];  //printf("id:%d -> pid:%d\n",id,pid);
            if(pid!=UINT_MAX){
              tfloat3 ps=Pos[pid];
              tfloat3 ps2=MatrixMulPoint(mvmatrix,ps);
              if(Simulate2D)ps2.y=ps.y;
              float dx=ps2.x-ps.x, dy=ps2.y-ps.y, dz=ps2.z-ps.z;
              if(abs(dx)>MovLimit||abs(dy)>MovLimit||abs(dz)>MovLimit)Code[pid]=CODE_SetOutMove(Code[pid]);
              Pos[pid]=ps2;
              Vel[pid]=TFloat3(dx/stepdt,dy/stepdt,dz/stepdt);
            }
          }
        }
      }
      BoundChanged=true;
    }
  }
  TmcStop(Timers,TMC_SuMotion);
}

//==============================================================================
/// Shows active timers.
//==============================================================================
void JSphCpu::ShowTimers(bool onlyfile){
  JLog2::TpMode_Out mode=(onlyfile? JLog2::Out_File: JLog2::Out_ScrFile);
  Log->Print("\n[CPU Timers]",mode);
  if(!SvTimers)Log->Print("none",mode);
  else for(unsigned c=0;c<TimerGetCount();c++)if(TimerIsActive(c))Log->Print(TimerToText(c),mode);
}

//==============================================================================
/// Returns string with name and values of active timers.
//==============================================================================
void JSphCpu::GetTimersInfo(std::string &hinfo,std::string &dinfo)const{
  for(unsigned c=0;c<TimerGetCount();c++)if(TimerIsActive(c)){
    hinfo=hinfo+";"+TimerGetName(c);
    dinfo=dinfo+";"+fun::FloatStr(TimerGetValue(c)/1000.f);
  }
}

#ifdef _WITHOMP
//==============================================================================
/// Summation of different blocks of array float to the first one.
//==============================================================================
void JSphCpu::OmpMergeDataSum(int ini,int fin,float *data,int stride,int nthreads){
  #pragma omp parallel for schedule (static)
  for(int c=ini;c<fin;c++){
    for(int t=1,c2=c+stride;t<nthreads;t++,c2+=stride)data[c]+=data[c2];
  }
}
//==============================================================================
/// Summation of different blocks of array tfloat3 to the first one.
//==============================================================================
void JSphCpu::OmpMergeDataSum(int ini,int fin,tfloat3 *data,int stride,int nthreads){
  #pragma omp parallel for schedule (static)
  for(int c=ini;c<fin;c++){
    for(int t=1,c2=c+stride;t<nthreads;t++,c2+=stride)data[c]=data[c]+data[c2];
  }
}
//==============================================================================
/// Summation of different blocks of array float to the first one when
/// the value is different from verror
//==============================================================================
void JSphCpu::OmpMergeDataSumError(int ini,int fin,float *data,int stride,int nthreads,float verror){
  #pragma omp parallel for schedule (static)
  for(int c=ini;c<fin;c++){
    for(int t=1,c2=c+stride;t<nthreads;t++,c2+=stride)data[c]=(data[c]==verror||data[c2]==verror? verror: data[c]+data[c2]);
  }
}
#endif








