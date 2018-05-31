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

/// \file JSphCpuSingle.cpp \brief Implements the class \ref JSphCpuSingle.

#include "JSphCpuSingle.h"
#include "Functions.h"
#include "JSphMotion.h"
#include "JCellDivCpuSingle.h"
#include "JPartsOut.h"
#include "JFormatFiles2.h"
#include "JPartsLoad.h"
#include "JXml.h"
#include "JPeriodicCpu.h"
#include <climits>

using namespace std;
//==============================================================================
/// Constructor.
//==============================================================================
JSphCpuSingle::JSphCpuSingle(){
  ClassName="JSphCpuSingle";
  CellDivSingle=NULL;
  PartsLoaded=NULL;
  PeriZone=NULL;
}

//==============================================================================
/// Destructor.
//==============================================================================
JSphCpuSingle::~JSphCpuSingle(){
  delete CellDivSingle; CellDivSingle=NULL;
  delete PartsLoaded;   PartsLoaded=NULL;
  delete PeriZone;      PeriZone=NULL;
}

//==============================================================================
/// Releases memory.
//==============================================================================
void JSphCpuSingle::FreeMemory(){
  FreeMemoryParticles();
  FreeMemoryAuxParticles();
}

//==============================================================================
/// Allocates memory.
//==============================================================================
void JSphCpuSingle::AllocMemory(){
  AllocMemoryParticles(CaseNp);
  AllocMemoryAuxParticles(CaseNp,CaseNp-CaseNpb);
}

//==============================================================================
/// Returns the allocated memory in CPU.
//==============================================================================
long long JSphCpuSingle::GetAllocMemoryCpu()const{  
  long long s=JSphCpu::GetAllocMemoryCpu();
  //-Allocated in other objects.
  if(CellDivSingle)s+=CellDivSingle->GetAllocMemory();
  if(PartsLoaded)s+=PartsLoaded->GetAllocMemory();
  if(PeriZone)s+=PeriZone->GetAllocMemory();
  return(s);
}

//==============================================================================
/// Updates the maximum values of memory, particles and cells.
//==============================================================================
void JSphCpuSingle::UpdateMaxValues(){
  MaxParticles=max(MaxParticles,Np);
  if(CellDivSingle)MaxCells=max(MaxCells,CellDivSingle->GetNct());
  long long m=GetAllocMemoryCpu();
  MaxMemoryCpu=max(MaxMemoryCpu,m);
}

//==============================================================================
/// Loads the configuration of the execution.
//==============================================================================
void JSphCpuSingle::LoadConfig(JCfgRun *cfg){
  ConfigOmp(cfg);            //-Loads OpenMp configuration.
  JSph::LoadConfig(cfg);     //-Loads general configuration.
}

//==============================================================================
/// Loads particles of the case to be processesed.
//==============================================================================
void JSphCpuSingle::LoadCaseParticles(){
  Log->Print("Loading initial state of particles...");
  getchar();
  printf('ok up JSphCpuSingle::LoadCaseParticles');
  PartsLoaded4=new JPartsLoad4;
  //PartsLoaded->LoadParticles(JPartsLoad::LOUT_AllOut,DirCase,CaseName,PartBegin,PartBeginDir,CaseNp,CaseNbound,CaseNfixed,CaseNmoving,CaseNfloat);
  
  PartsLoaded4->LoadParticles4(DirCase,CaseName,PartBegin,PartBeginDir);
  printf('ok up tohere1');
  PartsLoaded4->CheckConfig4(CaseNp,CaseNfixed,CaseNmoving,CaseNfloat,CaseNfluid,PeriX,PeriY,PeriZ);
  printf('ok up tohere2');
  getchar();
  PartsLoaded=new JPartsLoad;

  sprintf(Cad,"Loaded particles: %u (%u+%u)",PartsLoaded->GetCount(),PartsLoaded->GetCountOk(),PartsLoaded->GetCountOut()); Log->Print(Cad);
  SetOutInitialCount(PartsLoaded->GetCount()-PartsLoaded->GetCountOk());
  //-Recovers information of the loaded particles.
  Simulate2D=PartsLoaded->GetSimulate2D();
  if(Simulate2D&&PeriY)RunException("LoadCaseParticles","Periodic conditions in Y with 2D simulations can not be used.");
  const float bordermax=H*BORDER_MAP;
  const float peribordermax=Dp/2.f;
  const tfloat3 border=TFloat3((PeriX? peribordermax: bordermax),(PeriY? peribordermax: bordermax),(PeriZ? peribordermax: bordermax));
  
  PartsLoaded->GetLimits(border,border,MapPosMin,MapPosMax);
  
  if(!PartBegin||PartsLoaded->LoadPartBi2())ResizeMapLimits();
  if(PartBegin)PartBeginTimeStep=PartsLoaded->GetPartBeginTimeStep();
  sprintf(Cad,"MapPos=(%f,%f,%f)-(%f,%f,%f)",MapPosMin.x,MapPosMin.y,MapPosMin.z,MapPosMax.x,MapPosMax.y,MapPosMax.z); Log->Print(Cad);
  
  Log->Print("**Initial state of particles is loaded");
  //-Configures limits of periodic axis.
  if(PeriX)PeriXinc.x=MapPosMin.x-MapPosMax.x;
  if(PeriY)PeriYinc.y=MapPosMin.y-MapPosMax.y;
  if(PeriZ)PeriZinc.z=MapPosMin.z-MapPosMax.z;
}

//==============================================================================
/// Configuration of the current domain.
//==============================================================================
void JSphCpuSingle::ConfigDomain(){
  const char* met="ConfigDomain";
  //-Allocates memory in CPU and locates particle data in the variables of the domain.
  AllocMemory();
  //-Copies particle data.
  if(CaseNp!=PartsLoaded->GetCount())RunException(met,"The number of particles is invalid.");
  memcpy(Idp,PartsLoaded->GetIdp(),CaseNp*sizeof(unsigned));
  memcpy(Pos,PartsLoaded->GetPos(),CaseNp*sizeof(tfloat3));
  memcpy(Vel,PartsLoaded->GetVel(),CaseNp*sizeof(tfloat3));
  memcpy(Rhop,PartsLoaded->GetRhop(),CaseNp*sizeof(float));
  //-Loads code of particles.
  LoadCodeParticles(CaseNp,PartsLoaded->GetCountOut(),Idp,Code);
  //-Releases memory of PartsLoaded.
  delete PartsLoaded; PartsLoaded=NULL;
  //-Computes number of particles.
  Np=CaseNp; Npb=CaseNpb; NpbOk=Npb;
  //-Applies configuration of CellOrder.
  ConfigCellOrder(CellOrder,Np,Pos,Vel);

  //-Creates object for CellDiv and selects a valid cellmode.
  CellDivSingle=new JCellDivCpuSingle(Stable,Log,DirOut,PeriActive,TVisco==VISCO_LaminarSPS,!PeriActive,MapPosMin,MapPosMax,Dosh,CaseNbound,CaseNfixed,CaseNpb,CellOrder); //ALEX_SPS
  //-Selects a valid cellmode.
  if(CellMode==CELLMODE_None)CellMode=(Simulate2D? CELLMODE_2H: CELLMODE_H);
  CellDivSingle->ConfigInit(CellMode,CaseNp,CaseNpb,RhopOut,RhopOutMin,RhopOutMax);
  Hdiv=CellDivSingle->GetHdiv();
  Scell=CellDivSingle->GetScell();
  MapCells=CellDivSingle->GetMapCells();
  MovLimit=Scell*0.9f;
  Log->Print(fun::VarStr("CellMode",string(GetNameCellMode(CellMode))));
  Log->Print(fun::VarStr("Hdiv",Hdiv));
  Log->Print(string("MapCells=(")+fun::Uint3Str(OrderDecode(MapCells))+")");
  SaveMapCellsVtk(Scell);
  ConfigCellDiv((JCellDivCpu*)CellDivSingle);
  //-Initialises the use of periodic conditions.
  if(PeriActive)PeriInit();
  //-Reorders particles according to cells.
  BoundChanged=true;
  RunCellDivide(false);
}

//==============================================================================
/// Initialises the use of periodic conditions.
//==============================================================================
void JSphCpuSingle::PeriInit(){
  PeriZone=new JPeriodicCpu(Log,DirOut,MapPosMin,MapPosMax,(JCellDivCpu*)CellDivSingle);
  if(PeriX&&!PeriXY&&!PeriXZ)PeriZone->ConfigX(PeriXinc);
  if(PeriY&&!PeriXY&&!PeriYZ)PeriZone->ConfigY(PeriYinc);
  if(PeriZ&&!PeriXZ&&!PeriYZ)PeriZone->ConfigZ(PeriZinc);
  if(PeriXY)PeriZone->ConfigXY();
  if(PeriXZ)PeriZone->ConfigXZ();
  if(PeriYZ)PeriZone->ConfigYZ();
}

//==============================================================================
/// Checks position of particles for periodic conditions.
//==============================================================================
void JSphCpuSingle::PeriCheckPosition(){
  TmcStart(Timers,TMC_NlPeriCheck);
  const unsigned nzone=PeriZone->GetZonesCount();
  for(unsigned c=0;c<nzone;c++)PeriZone->CheckPosition(c,BoundChanged,Pos,Idp);
  TmcStop(Timers,TMC_NlPeriCheck);
}

//==============================================================================
/// Executes CellDiv of particles in cells.
//==============================================================================
void JSphCpuSingle::RunCellDivide(bool symvars){
  //-Checks positions of the particles in the periodic edges.
  if(PeriActive)PeriCheckPosition();
  //-Initiates Divide.
  CellDivSingle->Divide(BoundChanged,Idp,Pos,Rhop,Code,Timers);
  //-Orders particle data.
  TmcStart(Timers,TMC_NlSortData);
  CellDivSingle->SortParticles(Idp);
  CellDivSingle->SortParticles(Code);
  CellDivSingle->SortParticles(Pos);
  CellDivSingle->SortParticles(Vel);
  CellDivSingle->SortParticles(Rhop);
  if(TStep==STEP_Verlet){
    CellDivSingle->SortParticles(VelM1);
    CellDivSingle->SortParticles(RhopM1);
  }
  else if(TStep==STEP_Symplectic&&symvars){
    CellDivSingle->SortParticles(PosPre);
    CellDivSingle->SortParticles(VelPre); 
    CellDivSingle->SortParticles(RhopPre);
  }
  if(TVisco==VISCO_LaminarSPS){   
    CellDivSingle->SortParticles(Tau);
  }
  
  //-Recovers data of CellDiv and updates number of particles of each type.
  Np=CellDivSingle->GetNp();
  NpbOk=Npb-CellDivSingle->GetNpbIgnore();
  TmcStop(Timers,TMC_NlSortData);
  //-Manages excluded particles (bound and fluid).
  TmcStart(Timers,TMC_NlOutCheck);
  unsigned npfout=CellDivSingle->GetNpfOut();
  if(npfout){
    //-Reorders data of excluded particles (only fluid).
    if(CellOrder!=ORDER_XYZ){
      OrderDecodeData(CellOrder,npfout,Pos+Np);
      OrderDecodeData(CellOrder,npfout,Vel+Np);
    }
    PartsOut->AddParticles(npfout,Idp+Np,Pos+Np,Vel+Np,Rhop+Np,CellDivSingle->GetNpfOutRhop(),CellDivSingle->GetNpfOutMove());
  }
  TmcStop(Timers,TMC_NlOutCheck);
  BoundChanged=false;
}

//==============================================================================
/// Generates periodic zones and computes interaction.
//==============================================================================
void JSphCpuSingle::PeriInteraction(TpInter tinter){
  const unsigned nzone=PeriZone->GetZonesCount();
  for(unsigned c=0;c<nzone;c++){
    TmcStart(Timers,TMC_NlPeriPrepare); 
    PeriZone->PrepareForces(c,Pos,Rhop,Code,Idp,Vel,Csound,PrRhop,Tensil,Tau);
    TmcStop(Timers,TMC_NlPeriPrepare);
    TmcStart(Timers,TMC_CfPeriForces); 
    if(CaseNfloat){     //-Checks if there are floating particles in the periodic zone.
      const int np=PeriZone->GetListNp();
      const int npb=PeriZone->GetListNpb();
      const unsigned *list=PeriZone->GetList();
      const unsigned fluidini=PeriZone->GetListFluidIni();
      list=(list? list+npb: NULL);
      const int npf=np-npb;
      unsigned idfloat=UINT_MAX;
      for(int p=0;p<npf;p++){
        const unsigned p1=(list? list[p]: p+fluidini);     
        if(CODE_GetType(Code[p1])==CODE_TYPE_FLOATING)idfloat=min(idfloat,Idp[p1]);
      }
      if(idfloat!=UINT_MAX){
        sprintf(Cad,"Particle id:%u (from a floating body) in Periodicity region.",idfloat);
        RunException("PeriInteraction",Cad);
      }
    }
    InteractionPeri(tinter,PeriZone);
    TmcStop(Timers,TMC_CfPeriForces);
  }
  //RunException("PeriInteraction","Stop...");
}

//==============================================================================
/// Interaction for force computation.
//==============================================================================
void JSphCpuSingle::Interaction_Forces(TpInter tinter){
  PreInteraction_Forces(tinter);
  TmcStart(Timers,TMC_CfForces);
  InteractionCells(tinter);
  if(PeriActive){
    TmcStop(Timers,TMC_CfForces);
    PeriInteraction(tinter);
    TmcStart(Timers,TMC_CfForces);
  }
#ifdef _WITHOMP
  if(OmpMode!=OMPM_Single){
    OmpMergeDataSum(0,Np,Ar,Np,OmpThreads);
    OmpMergeDataSum(Npb,Np,Ace,Np,OmpThreads);
    if(tinter==INTER_Forces)OmpMergeDataSum(Npb,Np,VelXcor,Np,OmpThreads);
    if(Delta)OmpMergeDataSumError(Npb,Np,Delta,Np,OmpThreads,FLT_MAX);
  }
#endif
  if(Delta){       //-Applies Delta-SPH to Arg[].
    const int ini=int(Npb),fin=int(Np);
    #ifdef _WITHOMP
      #pragma omp parallel for schedule (static)
    #endif
    for(int p=ini;p<fin;p++)if(Delta[p]!=FLT_MAX)Ar[p]+=Delta[p];
  }

//  for(unsigned p=0;p<Np;p++)if(p==147){
//    sprintf(cad,"particle[%u]> idp:%u  ar:%f  ace:(%f,%f,%f)",p,Idp[p],Ar[p],Ace[p].x,Ace[p].y,Ace[p].z); Log->Print(cad);
//    sprintf(cad,"particle[%u]> idp:%u  ar:%f  vcor:(%f,%f,%f)",p,Idp[p],Ar[p],VelXcor[p].x,VelXcor[p].y,VelXcor[p].z); Log->Print(cad);
//  }

  if(TVisco==VISCO_LaminarSPS)SPSCalcTau(); //-Computes sub-particle stress tensor (Tau) for SPS turbulence model. 

  for(int t=0;t<OmpThreads;t++)ViscDtMax=max(ViscDtMax,ViscDtThread[t*STRIDE_OMP]);//-Reduction of ViscDt.
  if(Simulate2D)for(unsigned p=Npb;p<Np;p++)Ace[p].y=0;
  TmcStop(Timers,TMC_CfForces);
}

//==============================================================================
/// Computation of the step: Particle interaction and update of particle data
/// according to the forces computed in the interaction using VERLET.
//==============================================================================
float JSphCpuSingle::ComputeStep_Ver(bool rhopbound){
  Interaction_Forces(INTER_Forces);    //-Interaction.
  const float dt=DtVariable();         //-Computes new dt.
  ComputeVerlet(rhopbound,dt);         //-Updates particles using Verlet.
  if(CaseNfloat)RunFloating(dt,false); //-Manages floating bodies.
  return(dt);
}

//==============================================================================
/// Computation of the step: Particle interaction and update of particle data
/// according to the forces computed in the interaction using SYMPLECTIC.
//==============================================================================
float JSphCpuSingle::ComputeStep_Sym(bool rhopbound){
  const float dt=DtPre;
  //-Predictor.
  //-----------
  Interaction_Forces(INTER_Forces);         //-Interaction.
  const float ddt_p=DtVariable();           //-Computes dt of Predictor.
  ComputeSymplecticPre(rhopbound,dt);       //-Applies Symplectic-Predictor to particle data.
  if(CaseNfloat)RunFloating(dt*0.5f,true);  //-Processes floating bodies.
  //-Corrector.
  //-----------
  RunCellDivide(true);
  Interaction_Forces(INTER_ForcesCorr);     //-Interaction without VelXCor[].
  ComputeSymplecticCorr(rhopbound,dt);      //-Applies Symplectic-Corrector to particle data.
  if(CaseNfloat)RunFloating(dt,false);      //-Processes floating bodies.
  //-Computes dt for the following ComputeStep.
  DtPre=min(ddt_p,DtVariable());
  return(dt);
}

//==============================================================================
/// Processes floating bodies.
//==============================================================================
void JSphCpuSingle::RunFloating(float dt2,bool predictor){
  if(TimeStep>FtPause){
    TmcStart(Timers,TMC_SuFloating);
    CalcFtRidp(); 
    for(unsigned cf=0;cf<FtCount;cf++){
      StFloatingData *fobj=FtObjs+cf;
      //-Computes traslational and rotational velocities.
      tfloat3 face=TFloat3(0);
      tfloat3 fomegavel=TFloat3(0);
      const unsigned fpini=fobj->begin-CaseNpb;
      const unsigned fpfin=fpini+fobj->count;
      for(unsigned fp=fpini;fp<fpfin;fp++){
        int p=FtRidp[fp];
        //-Ace is initialised with the value of the gravity for all particles.
        float acex=Ace[p].x-Gravity.x,acey=Ace[p].y-Gravity.y,acez=Ace[p].z-Gravity.z;
        face.x+=acex; face.y+=acey; face.z+=acez;
        tfloat3 dist=FtDist[fp];
        fomegavel.x+= acez*dist.y - acey*dist.z;
        fomegavel.y+= acex*dist.z - acez*dist.x;
        fomegavel.z+= acey*dist.x - acex*dist.y;
      }
      face.x=(face.x+fobj->mass*Gravity.x)/fobj->mass;
      face.y=(face.y+fobj->mass*Gravity.y)/fobj->mass;
      face.z=(face.z+fobj->mass*Gravity.z)/fobj->mass;
      //sprintf(Cad,"%u>> face:%s fomegavel:%s",Nstep,fun::Float3Str(face).c_str(),fun::Float3Str(fomegavel).c_str()); Log->PrintDbg(Cad);
      //-Recomputes values of floating.
      tfloat3 center=fobj->center;
      tfloat3 fvel=fobj->fvel;
      tfloat3 fomega=fobj->fomega;
      fomegavel.x/=fobj->inertia.x;
      fomegavel.y/=fobj->inertia.y;
      fomegavel.z/=fobj->inertia.z;
      if(Simulate2D){ face.y=0; fomegavel.x=0; fomegavel.z=0; fvel.y=0; }
      center.x+=dt2*fvel.x;
      center.y+=dt2*fvel.y;
      center.z+=dt2*fvel.z;
      fvel.x+=dt2*face.x;
      fvel.y+=dt2*face.y;
      fvel.z+=dt2*face.z;
      fomega.x+=dt2*fomegavel.x;
      fomega.y+=dt2*fomegavel.y;
      fomega.z+=dt2*fomegavel.z;
      //-Updates floating particles.
      if(Simulate2D)for(unsigned fp=fpini;fp<fpfin;fp++){
        int p=FtRidp[fp];
        tfloat3 *pos=Pos+p,*vel=Vel+p;
        pos->x+=dt2*vel->x;  pos->z+=dt2*vel->z;
        tfloat3 distaux;
        tfloat3 *dist=(predictor? &distaux: FtDist+fp); 
        *dist=TFloat3(pos->x-center.x,0,pos->z-center.z); 
        vel->x=fvel.x+(fomega.y*dist->z-fomega.z*dist->y);
        vel->y=0;
        vel->z=fvel.z+(fomega.x*dist->y-fomega.y*dist->x);
      }
      else for(unsigned fp=fpini;fp<fpfin;fp++){
        int p=FtRidp[fp];
        tfloat3 *pos=Pos+p,*vel=Vel+p;
        pos->x+=dt2*vel->x;  pos->y+=dt2*vel->y;  pos->z+=dt2*vel->z;
        tfloat3 distaux;
        tfloat3 *dist=(predictor? &distaux: FtDist+fp); 
        *dist=TFloat3(pos->x-center.x,pos->y-center.y,pos->z-center.z);   
        vel->x=fvel.x+(fomega.y*dist->z-fomega.z*dist->y);
        vel->y=fvel.y+(fomega.z*dist->x-fomega.x*dist->z);
        vel->z=fvel.z+(fomega.x*dist->y-fomega.y*dist->x);
      }
      //-Stores data.
      if(!predictor){
        fobj->center=center;
        fobj->fvel=fvel;
        fobj->fomega=fomega;
        //sprintf(Cad,"-----<%u> center:%f,%f,%f  fvel:%f,%f,%f  fomega:%f,%f,%f",Nstep,center.x,center.y,center.z,fvel.x,fvel.y,fvel.z,fomega.x,fomega.y,fomega.z); Log->PrintDbg(Cad);
      }
    }
    TmcStop(Timers,TMC_SuFloating);
  }
}

//==============================================================================
/// Generates periodic zones of interaction and computes Shepard interaction.
//==============================================================================
void JSphCpuSingle::PeriInteractionShepard(){
  const unsigned nzone=PeriZone->GetZonesCount();
  for(unsigned c=0;c<nzone;c++){
    TmcStart(Timers,TMC_NlPeriPrepare); 
    PeriZone->PrepareShepard(c,Pos,Rhop,Code);
    TmcStop(Timers,TMC_NlPeriPrepare);
    TmcStart(Timers,TMC_CfPeriForces);
    InteractionPeri(INTER_Shepard,PeriZone);
    TmcStop(Timers,TMC_CfPeriForces);
  }
}

//==============================================================================
/// Applies Shepard density filter.
//==============================================================================
void JSphCpuSingle::RunShepard(){
  TmcStart(Timers,TMC_CfShepard);
  PreInteraction_Shepard();
  InteractionCells(INTER_Shepard);
  if(PeriActive){
    TmcStop(Timers,TMC_CfShepard);
    PeriInteractionShepard();
    TmcStart(Timers,TMC_CfShepard);
  }
  const int npf=Np-Npb;
  #ifdef _WITHOMP
    if(OmpMode!=OMPM_Single){
      OmpMergeDataSum(0,npf,FdWab,npf,OmpThreads);
      OmpMergeDataSum(0,npf,FdRhop,npf,OmpThreads);
    }
  #endif
  ComputeShepard(Npb,Np);
  TmcStop(Timers,TMC_CfShepard);
}

//==============================================================================
/// Runs the simulation.
//==============================================================================
void JSphCpuSingle::Run(std::string appname,JCfgRun *cfg,JLog2 *log){
  const char* met="Run";
  if(!cfg||!log)return;
  AppName=appname; Log=log;

  //-Configures timers.
  //-------------------
  TmcCreation(Timers,cfg->SvTimers);
  TmcStart(Timers,TMC_Init);

  //-Loads parameters and input data.
  //-----------------------------------------
  LoadConfig(cfg);
  LoadCaseParticles();
  ConfigConstants(Simulate2D);
  ConfigDomain();
  ConfigRunMode(cfg);

//-Initialisation of variables of execution.
  //-------------------------------------------
  InitRun();
  UpdateMaxValues();
  PrintAllocMemory(GetAllocMemoryCpu());
  TmcStop(Timers,TMC_Init);
  SaveData();
  PartNstep=-1; Part++;

  //-MAIN LOOP.
  //------------------
  bool partoutstop=false;
  TimerSim.Start();
  TimerPart.Start();
  Log->Print(string("\n[Initialising simulation (")+RunCode+")  "+fun::GetDateTime()+"]");
  PrintHeadPart();
  while(TimeStep<TimeMax){
    float stepdt=ComputeStep(true);
    if(PartDtMin>stepdt)PartDtMin=stepdt; if(PartDtMax<stepdt)PartDtMax=stepdt;
    if(CaseNmoving)RunMotion(stepdt);
    RunCellDivide(false);
    if(ShepardSteps&&Nstep&&( !((Nstep+1)%ShepardSteps) || (TStep==STEP_Verlet && !(Nstep%ShepardSteps)) ))RunShepard(); //-Shepard density filter.
    //-Verlet+Shepard: Combining Shepar with Verlet, 
    // Verlet must be applied in the corresponding step and also in the next one
    // to smooth Rhop & RhopM1.
    TimeStep+=stepdt;
    partoutstop=unsigned(CaseNp-Np)>=PartOutMax;
    if(TimeStep-TimePart*(Part-1)>=TimePart||SvSteps||partoutstop){
      if(partoutstop){
        Log->Print("\n**** Particles OUT limit reached...\n");
        TimeMax=TimeStep;
      }
      SaveData();
      Part++;
      PartNstep=Nstep;
      TimeStepM1=TimeStep;
      TimerPart.Start();
    }
    UpdateMaxValues();
    Nstep++;
    //if(Nstep>=2280)break;
  }
  TimerSim.Stop(); TimerTot.Stop();

  //-End of simulation.
  //--------------------
  FinishRun(partoutstop);
}

//==============================================================================
/// Generates ouput files of particle data.
//==============================================================================
void JSphCpuSingle::SaveData(){
  const bool save=(SvData!=SDAT_None);
  TmcStart(Timers,TMC_SuSavePart);
  //-Reorder particle data in the original order.
  if(save&&CellOrder!=ORDER_XYZ){
    OrderDecodeData(CellOrder,Np,Pos);
    OrderDecodeData(CellOrder,Np,Vel);
  }
  //-Recovers data of excluded particles.
  tfloat3 vdom[2]={OrderDecode(CellDivSingle->GetDomainLimits(true)),OrderDecode(CellDivSingle->GetDomainLimits(false))};
  unsigned noutpos=PartsOut->GetOutPosCount(),noutrhop=PartsOut->GetOutRhopCount(),noutmove=PartsOut->GetOutMoveCount();
  unsigned nout=noutpos+noutrhop+noutmove;
  if(save){
    memcpy(Idp+Np,PartsOut->GetIdpOut(),sizeof(unsigned)*nout);
    memcpy(Pos+Np,PartsOut->GetPosOut(),sizeof(tfloat3)*nout);
    memcpy(Vel+Np,PartsOut->GetVelOut(),sizeof(tfloat3)*nout);
    memcpy(Rhop+Np,PartsOut->GetRhopOut(),sizeof(float)*nout);
  }
  PartsOut->Clear();
  //-Stores particle data.
  //if(SvData&SDAT_Info){
  //  StInfoPartPlus infoplus;
  //  infoplus.nct=CellDivSingle->GetNct();
  //  infoplus.npbin=NpbOk;
  //  infoplus.npbout=Npb-NpbOk;
  //  infoplus.npf=Np-Npb;
  //  infoplus.memorycpualloc=this->GetAllocMemoryCpu();
  //  infoplus.gpudata=false;
  //  TimerSim.Stop();
  //  infoplus.timesim=TimerSim.GetElapsedTimeF()/1000.f;
  //  SaveParData(Np,nout,Idp,Pos,Vel,Rhop,vdom[0],vdom[1],&infoplus);
  //}
  JSph::SaveData(Np,noutpos,noutrhop,noutmove,Idp,Pos,Vel,Rhop,NULL,NULL,NULL,NULL,1,vdom);
  //-Reorders particle data to the order of execution.
  if(save&&CellOrder!=ORDER_XYZ){
    OrderCodeData(CellOrder,Np,Pos);
    OrderCodeData(CellOrder,Np,Vel);
  }
  TmcStop(Timers,TMC_SuSavePart);
}

//==============================================================================
/// Displays and stores final brief of execution.
//==============================================================================
void JSphCpuSingle::FinishRun(bool stop){
  float tsim=TimerSim.GetElapsedTimeF()/1000.f,ttot=TimerTot.GetElapsedTimeF()/1000.f;
  JSph::ShowResume(stop,tsim,ttot,true,"");
  string hinfo=";RunMode",dinfo=string(";")+RunMode;
  if(SvTimers){
    ShowTimers();
    GetTimersInfo(hinfo,dinfo);
  }
  Log->Print(" ");
  if(SvRes)SaveRes(tsim,ttot,hinfo,dinfo);
  if(CaseNfloat)SaveFloatingDataTotal();
}


