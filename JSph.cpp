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

/// \file JSph.cpp \brief Implements the class \ref JSph

#include "JSph.h"
#include "Functions.h"
#include "JSphMotion.h"
#include "JXml.h"
#include "JSpaceCtes.h"
#include "JSpaceEParms.h"
#include "JSpaceParts.h"
#include "JFormatFiles2.h"
#include "JCellDivCpu.h"
#include "JFormatFiles2.h"
#include "JPartData.h"
#include "JFloatingData.h"
#include "JSphDtFixed.h"
#include "JSphVarAcc.h"

#include "JPartDataBi4.h"
#include "JPartOutBi4Save.h"
#include "JPartFloatBi4.h"
#include "JPartsOut.h"
#include <climits>

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JSph::JSph(){
  ClassName="JSph";
  TStep=STEP_None;
  BinData=NULL;
  DataBi4=NULL;
  FtData=NULL;
  DtFixed=NULL;
  VarAcc=NULL;
  MkList=NULL;
  FtObjs=NULL;
  Log=NULL;

  //multi
  PhaseCte=NULL;
  PhaseArray=NULL;

  FtFileData="FloatingData.dat";
  Motion=new JSphMotion();
  InitVars();
}

//==============================================================================
/// Destructor.
//==============================================================================
JSph::~JSph(){
  delete BinData;
  delete DataBi4;
  delete FtData;
  delete DtFixed;
  delete VarAcc;
  //multi
  delete[] PhaseCte;
  delete[] PhaseArray;
  
  ResetMkInfo();
  AllocMemoryFloating(0);
  delete Motion;
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSph::InitVars(){
  Log=NULL;
  Simulate2D=false;
  Hdiv=0;
  VerletSteps=40;
  ShepardSteps=30;
  TDeltaSph=DELTA_None; DeltaSph=0;
  memset(&CubicCte,0,sizeof(StCubicCte));
  memset(&WendlandCte,0,sizeof(StWendlandCte));
  //memsets for multiphase ararys in int multi
  CaseName="";
  RunName="";
  DirCase="";
  DirOut="";
  PartBegin=0; PartBeginDir=""; PartBeginFirst=0;
  PartBeginTimeStep=0;
  CaseNp=CaseNbound=CaseNfixed=CaseNmoving=CaseNfloat=CaseNfluid=CaseNpb=0;
  
  CasePosMin=CasePosMax=TDouble3(0);

  MapPosMin=MapPosMax=TFloat3(0);
  MapCells=TUint3(0);
  Part=PartIni=0; Nstep=0;
  TimeStepIni=0;
  DtModif=0;
  
  //SvData=byte(SDAT_Binx);
  SvData=byte(SDAT_Binx)|byte(SDAT_Info);
  
  SvSteps=false;
  SvDt=false;
  SvRes=false;
  SvTimers=false;
  SvDomainVtk=false;
  MotionTimeMod=0;
  MotionObjCount=0;
  FtCount=0;
  FtPause=0;
  RhopOut=true; RhopOutMin=100; RhopOutMax=3000;
  ClearCfgDomain();
  OutPosCount=OutRhopCount=OutMoveCount=OutInitialCount=0;
  ResetMkInfo();
  AllocMemoryFloating(0);
  MaxMemoryCpu=MaxMemoryGpu=MaxParticles=MaxCells=0;
  PartDtMin=FLT_MAX; PartDtMax=-FLT_MAX;
  RunCode=CalcRunCode();
  PeriActive=0;
  PeriX=PeriY=PeriZ=PeriXY=PeriXZ=PeriYZ=false;
  PeriXinc=PeriYinc=PeriZinc=TFloat3(0);
}

//==============================================================================
/// Generates a random code to identify the file of the results of the execution.
//==============================================================================
std::string JSph::CalcRunCode()const{
  srand((unsigned)time(NULL));
  const unsigned len=8;
  char code[len+1];
  for(unsigned c=0;c<len;c++){
    char let=char(float(rand())/float(RAND_MAX)*36);
    code[c]=(let<10? let+48: let+87);
  } 
  code[len]=0;
  return(code);
}

//==============================================================================
/// Returns the code version in text format.
//==============================================================================
std::string JSph::GetVersionStr(){
  char cad[128];
  sprintf(cad,"%1.2f",float(VersionMajor)/100);
  //sprintf(cad,"%5.2f.%d",float(VersionMajor)/100,VersionMinor);
  return(cad);
}
//==============================================================================
/// Sets the configuration of the domain limits by default.
//==============================================================================
void JSph::ClearCfgDomain(){
  CfgDomainParticles=true;
  CfgDomainParticlesMin=CfgDomainParticlesMax=TFloat3(0);
  CfgDomainParticlesPrcMin=CfgDomainParticlesPrcMax=TFloat3(0);
  CfgDomainFixedMin=CfgDomainFixedMax=TFloat3(0);
}

//==============================================================================
/// Sets the configuration of the domain limits using given values.
//==============================================================================
void JSph::ConfigDomainFixed(tfloat3 vmin,tfloat3 vmax){
  ClearCfgDomain();
  CfgDomainParticles=false;
  CfgDomainFixedMin=vmin; CfgDomainFixedMax=vmax;
}
//==============================================================================
/// Sets the configuration of the domain limits using positions of particles.
//==============================================================================
void JSph::ConfigDomainParticles(tfloat3 vmin,tfloat3 vmax){
  CfgDomainParticles=true;
  CfgDomainParticlesMin=vmin; CfgDomainParticlesMax=vmax;
}
//==============================================================================
/// Sets the configuration of the domain limits using positions plus a percentage.
//==============================================================================
void JSph::ConfigDomainParticlesPrc(tfloat3 vmin,tfloat3 vmax){
  CfgDomainParticles=true;
  CfgDomainParticlesPrcMin=vmin; CfgDomainParticlesPrcMax=vmax;
}

//==============================================================================
/// Allocates memory of floating objectcs.
//==============================================================================
void JSph::AllocMemoryFloating(unsigned ftcount){
  delete[] FtObjs; FtObjs=NULL;
  if(ftcount)FtObjs=new StFloatingData[ftcount];
}

//==============================================================================
/// Returns the allocated memory in CPU.
//==============================================================================
long long JSph::GetAllocMemoryCpu()const{  
  //-Allocated in AllocMemoryCase().
  long long s=0;
  //-Allocated in AllocMemoryFloating().
  if(FtObjs)s+=FtCount*sizeof(StFloatingData);
  //-Allocated in other objects.
  if(BinData)s+=BinData->GetMemoryAlloc();
  
  //if(DataBi4)s+=DataBi4->GetMemoryAlloc();
  
  if(DtFixed)s+=DtFixed->GetAllocMemory();
  if(VarAcc)s+=VarAcc->GetAllocMemory();
  return(s);
}

//==============================================================================
/// Loads the configuration of the execution.
//==============================================================================
void JSph::LoadConfig(const JCfgRun *cfg){
  const char* met="LoadConfig";
  TimerTot.Start();
  Stable=cfg->Stable;
  DirOut=fun::GetDirWithSlash(cfg->DirOut);
  CaseName=cfg->CaseName; 
  DirCase=fun::GetDirWithSlash(fun::GetDirParent(CaseName));
  CaseName=CaseName.substr(DirCase.length());
  if(!CaseName.length())RunException(met,"Name of the case for execution was not indicated.");
  RunName=(cfg->RunName.length()? cfg->RunName: CaseName);
  PartBeginDir=cfg->PartBeginDir; PartBegin=cfg->PartBegin; PartBeginFirst=cfg->PartBeginFirst;

  //-Output options:
  SvData=byte(SDAT_None); 
  if(cfg->Sv_Csv)SvData|=byte(SDAT_Csv);
  if(cfg->Sv_Binx)SvData|=byte(SDAT_Binx);
  if(cfg->Sv_Info)SvData|=byte(SDAT_Info);
  if(cfg->Sv_Vtk)SvData|=byte(SDAT_Vtk);

  SvRes=cfg->SvRes;
  SvTimers=cfg->SvTimers;
  SvDomainVtk=cfg->SvDomainVtk;
  SvSteps=false;                   //-Stores all steps.

  printf("\n");
  char cad[256];
  RunTimeDate=fun::GetDateTime();
  sprintf(cad,"[Initialising %s v%s  %s]",ClassName.c_str(),GetVersionStr().c_str(),RunTimeDate.c_str());
  Log->Print(cad);

  string tx=fun::VarStr("CaseName",CaseName);
  tx=tx+";\n"+fun::VarStr("DirCase",DirCase)+";\n"+fun::VarStr("RunName",RunName)+";\n"+fun::VarStr("DirOut",DirOut)+";";
  if(PartBegin){
    Log->Print(fun::VarStr("PartBegin",PartBegin));
    Log->Print(fun::VarStr("PartBeginDir",PartBeginDir));
    Log->Print(fun::VarStr("PartBeginFirst",PartBeginFirst));
  }

  LoadCaseConfig();

  //-Aplies configuration using command line.
  if(cfg->TStep)TStep=cfg->TStep;
  if(cfg->VerletSteps>=0)VerletSteps=cfg->VerletSteps;
  if(cfg->TKernel)TKernel=cfg->TKernel;
  if(cfg->TVisco){ TVisco=cfg->TVisco; Visco=cfg->Visco; }
  if(cfg->ShepardSteps>=0)ShepardSteps=cfg->ShepardSteps;
  if(cfg->DeltaSph>=0){
    DeltaSph=cfg->DeltaSph;
    TDeltaSph=(DeltaSph? DELTA_DBC: DELTA_None);
  }
  if(TDeltaSph==DELTA_DBC&&(PeriActive||Cpu))TDeltaSph=DELTA_DBCExt;
  if(cfg->TimeMax>0)TimeMax=cfg->TimeMax;
  if(cfg->TimePart>=0)TimePart=cfg->TimePart;
  if(cfg->FtPause>=0)FtPause=cfg->FtPause;

  CellOrder=cfg->CellOrder;
  CellMode=cfg->CellMode;
  if(cfg->DomainMode==1){
    ConfigDomainParticles(cfg->DomainParticlesMin,cfg->DomainParticlesMax);
    ConfigDomainParticlesPrc(cfg->DomainParticlesPrcMin,cfg->DomainParticlesPrcMax);
  }
  else if(cfg->DomainMode==2)ConfigDomainFixed(cfg->DomainFixedMin,cfg->DomainFixedMax);
  if(cfg->RhopOutModif){ 
    RhopOutMin=cfg->RhopOutMin; RhopOutMax=cfg->RhopOutMax;
  }
  RhopOut=(RhopOutMin<RhopOutMax);
  if(!RhopOut){ RhopOutMin=-FLT_MAX; RhopOutMax=FLT_MAX; }

  if(Cpu)RunException(met,"CPU execution is not available.");
  if(CaseNfloat)RunException(met,"Execution with floating bodies is not available.");
  if(PeriActive)RunException(met,"Execution with periodic boundary conditions is not available.");
  if(TStep!=STEP_Symplectic)RunException(met,"Only execution with Symplectic is available.");
  if(ShepardSteps!=0)RunException(met,"Execution with Shepard is not available.");
  if(TKernel!=KERNEL_Wendland)RunException(met,"Only execution with kernel Wendland is available.");
  if(TVisco!=VISCO_SumSPS)RunException(met,"Only execution with arbitary viscosiy formulation is available (using constitutive equations).");
}

//==============================================================================
/// Loads the case configuration to be executed.
//==============================================================================
void JSph::LoadCaseConfig(){
  const char* met="LoadCaseConfig";
  string filexml=DirCase+CaseName+".xml";
  if(!fun::FileExists(filexml))RunException(met,"Case configuration was not found.",filexml);
  JXml xml; xml.LoadFile(filexml);
  JSpaceCtes ctes;     ctes.LoadXmlRun(&xml,"case.execution.constants");
  JSpaceEParms eparms; eparms.LoadXml(&xml,"case.execution.parameters");
  JSpaceParts parts;    parts.LoadXml(&xml,"case.execution.particles");

  //-Execution parameters.
  switch(eparms.GetValueInt("StepAlgorithm",true,2)){
    case 1:  TStep=STEP_Verlet;     break;
    case 2:  TStep=STEP_Symplectic;  break;
    default: RunException(met,"Step algorithm is not valid.");
  }
  VerletSteps=eparms.GetValueInt("VerletSteps",true,40);
  switch(eparms.GetValueInt("Kernel",true,2)){
    case 1:  TKernel=KERNEL_Cubic;     break;
    case 2:  TKernel=KERNEL_Wendland;  break;
    default: RunException(met,"Kernel choice is not valid.");
  }
  switch(eparms.GetValueInt("ViscoTreatment",true,3)){
    case 1:  TVisco=VISCO_Artificial;  break;
    case 2:  TVisco=VISCO_LaminarSPS;  break;  
	case 3:  TVisco=VISCO_SumSPS;  break;  
    default: RunException(met,"Viscosity treatment is not valid.");
  }
  Visco=eparms.GetValueFloat("Visco",true,0.0001f);
  ShepardSteps=eparms.GetValueInt("ShepardSteps",true,0);
  DeltaSph=eparms.GetValueFloat("DeltaSPH",true,0);
  TDeltaSph=(DeltaSph? DELTA_DBC: DELTA_None);
  FtPause=eparms.GetValueFloat("FtPause",true,0);
  TimeMax=eparms.GetValueFloat("TimeMax");
  TimePart=eparms.GetValueFloat("TimeOut");
  DtIni=eparms.GetValueFloat("DtIni");
  DtMin=eparms.GetValueFloat("DtMin",true,0.00001f);
  string filedtfixed=eparms.GetValueStr("DtFixed",true);
  if(!filedtfixed.empty()){
    DtFixed=new JSphDtFixed();
    if(int(filedtfixed.find("/"))<0&&int(filedtfixed.find("\\"))<0)filedtfixed=DirCase+filedtfixed; //-Only name of the file.
    DtFixed->LoadFile(filedtfixed);
  }
  if(eparms.Exists("RhopOutMin"))RhopOutMin=eparms.GetValueFloat("RhopOutMin");
  if(eparms.Exists("RhopOutMax"))RhopOutMax=eparms.GetValueFloat("RhopOutMax");

  //-Configuration of periodic boundaries.
  if(eparms.Exists("XPeriodicIncY")){ PeriXinc.y=eparms.GetValueFloat("XPeriodicIncY"); PeriX=true; }
  if(eparms.Exists("XPeriodicIncZ")){ PeriXinc.z=eparms.GetValueFloat("XPeriodicIncZ"); PeriX=true; }
  if(eparms.Exists("YPeriodicIncX")){ PeriYinc.x=eparms.GetValueFloat("YPeriodicIncX"); PeriY=true; }
  if(eparms.Exists("YPeriodicIncZ")){ PeriYinc.z=eparms.GetValueFloat("YPeriodicIncZ"); PeriY=true; }
  if(eparms.Exists("ZPeriodicIncX")){ PeriZinc.x=eparms.GetValueFloat("ZPeriodicIncX"); PeriZ=true; }
  if(eparms.Exists("ZPeriodicIncY")){ PeriZinc.y=eparms.GetValueFloat("ZPeriodicIncY"); PeriZ=true; }
  if(eparms.Exists("XYPeriodic")){ PeriXY=PeriX=PeriY=true; PeriXZ=PeriYZ=false; PeriXinc=PeriYinc=TFloat3(0); }
  if(eparms.Exists("XZPeriodic")){ PeriXZ=PeriX=PeriZ=true; PeriXY=PeriYZ=false; PeriXinc=PeriZinc=TFloat3(0); }
  if(eparms.Exists("YZPeriodic")){ PeriYZ=PeriY=PeriZ=true; PeriXY=PeriXZ=false; PeriYinc=PeriZinc=TFloat3(0); }
  PeriActive=(PeriX? 1: 0)+(PeriY? 2: 0)+(PeriZ? 4: 0);

  //-Configuration of domain size.
  float incz=eparms.GetValueFloat("IncZ",true,0.f);
  if(incz){
    ClearCfgDomain();
    CfgDomainParticlesPrcMax.z=incz;
  }
  if(eparms.Exists("DomainParticles")){
    string key="DomainParticles";
    ConfigDomainParticles(TFloat3(eparms.GetValueNumFloat(key,0),eparms.GetValueNumFloat(key,1),eparms.GetValueNumFloat(key,2)),TFloat3(eparms.GetValueNumFloat(key,3),eparms.GetValueNumFloat(key,4),eparms.GetValueNumFloat(key,5)));
  }
  if(eparms.Exists("DomainParticlesPrc")){
    string key="DomainParticlesPrc";
    ConfigDomainParticlesPrc(TFloat3(eparms.GetValueNumFloat(key,0),eparms.GetValueNumFloat(key,1),eparms.GetValueNumFloat(key,2)),TFloat3(eparms.GetValueNumFloat(key,3),eparms.GetValueNumFloat(key,4),eparms.GetValueNumFloat(key,5)));
  }
  if(eparms.Exists("DomainFixed")){
    string key="DomainFixed";
    ConfigDomainFixed(TFloat3(eparms.GetValueNumFloat(key,0),eparms.GetValueNumFloat(key,1),eparms.GetValueNumFloat(key,2)),TFloat3(eparms.GetValueNumFloat(key,3),eparms.GetValueNumFloat(key,4),eparms.GetValueNumFloat(key,5)));
  }

  //-Reads the values for the base file path and file count for the variable acceleration input file(s)
  string accinput=eparms.GetValueStr("VarAccInput",true,"");
  int accinputcount=eparms.GetValueInt("VarAccInputCount",true,0);
  if(!accinput.empty() && accinputcount>0){
    VarAcc=new JSphVarAcc();
    if(int(accinput.find("/"))<0&&int(accinput.find("\\"))<0)accinput=DirCase+accinput; //-Only name of the file.
    VarAcc->Config(accinput,unsigned(accinputcount),TimeMax);
    Log->Print("Variable acceleration data successfully loaded.");
  }

  //-Predefined constantes.
  H=ctes.GetH();
  CteB=ctes.GetB();
  Gamma=ctes.GetGamma();
  Eps=ctes.GetEps();
  CFLnumber=ctes.GetCFLnumber();
  Dp=ctes.GetDp();
  Gravity=ctes.GetGravity();
  MassFluid=ctes.GetMassFluid();
  MassBound=ctes.GetMassBound();
  Rhop0=ctes.GetRhop0();
  OverRhop0=1.0f/Rhop0; 

  //-Particle data.
  CaseNp=parts.Count();
  CaseNfixed=parts.Count(PT_Fixed);
  CaseNmoving=parts.Count(PT_Moving);
  CaseNfloat=parts.Count(PT_Floating);
  CaseNfluid=parts.Count(PT_Fluid);
  CaseNbound=CaseNp-CaseNfluid;
  CaseNpb=CaseNbound-CaseNfloat;
  PartOutMax=unsigned(float(CaseNfluid)*eparms.GetValueFloat("PartsOutMax",true,1));

  NpDynamic=ReuseIds=false;
  TotalNp=CaseNp;

  //Read multi-phase variables 
  InitMultiPhase();
  
  //-Loads and configures MK of particles.
  LoadMkInfo(&parts);

  //-Loads and configures MOTION.
  MotionObjCount=0;
  for(unsigned c=0;c<parts.CountBlocks();c++){
    const JSpacePartBlock &block=parts.GetBlock(c);
    if(block.Type==PT_Moving){
      if(MotionObjCount>=255)RunException(met,"The number of mobile objects exceeds the maximum.");
      //printf("block[%2d]=%d -> %d\n",c,block.GetBegin(),block.GetCount());
      MotionObjBegin[MotionObjCount]=block.GetBegin();
      MotionObjBegin[MotionObjCount+1]=MotionObjBegin[MotionObjCount]+block.GetCount();
      MotionObjCount++;
    }
  }
  if(int(MotionObjCount)<Motion->Init(&xml,"case.execution.motion",DirCase))RunException(met,"The number of mobile objects is lower than expected.");

  //-Loads floating objects.
  FtCount=parts.CountBlocks(PT_Floating);
  if(FtCount){
    AllocMemoryFloating(FtCount);
    unsigned cobj=0;
    for(unsigned c=0;c<parts.CountBlocks()&&cobj<FtCount;c++){
      const JSpacePartBlock &block=parts.GetBlock(c);
      if(block.Type==PT_Floating){
        const JSpacePartBlock_Floating &fblock=(const JSpacePartBlock_Floating &)block;
        StFloatingData* fobj=FtObjs+cobj;
        fobj->begin=fblock.GetBegin();
        fobj->count=fblock.GetCount();
        fobj->mass=fblock.GetMassbody();
        fobj->massp=fobj->mass/fobj->count;
      //{ char cad[1024]; sprintf(cad,"++++> massp[%u]:%f",cobj,fobj->massp); Log->PrintDbg(cad); }
        fobj->center=fblock.GetCenter();
        fobj->inertia=fblock.GetInertia();
        fobj->fvel=fblock.GetVelini();
        fobj->fomega=fblock.GetOmegaini();
        cobj++;
      }
    }
  }
  Log->Print("**Case configuration is loaded");
}

//==============================================================================
// Load related data and adjusts Multi-phase fluid density.
//==============================================================================
void JSph::InitMultiPhase(){
  const char* met="InitMultiPhase";
  Log->Print("");
  Log->Print("[Multi-phase configuration]\n");
  string filexml=DirCase+CaseName+".xml";
  if(fun::FileType(filexml)!=2)RunException(met,"Case configuration was not found.",filexml);
  
  JXml xml; xml.LoadFile(filexml);
  JSpaceEParms eparms; eparms.LoadXml(&xml,"case.execution.parameters");
  JSpaceParts parts;    parts.LoadXml(&xml,"case.execution.particles");
  
  //-Load multi-phase configuration
  int PhaseCount=eparms.GetValueInt("PhaseCount",false);  // read phasecount from xml
  Log->Print(fun::VarStr("PhaseCount",PhaseCount));
  if(PhaseCount!=2)RunException(met,"Only execution with two phases is available.");
    
  PhaseCte=new StPhaseCte[PhaseCount];
  PhaseArray=new StPhaseArray[PhaseCount];
  
  for(int c=0;c<PhaseCount;c++){ //Two phases max for now
    string suffix=string("_")+fun::IntStr(c);
	byte mkfluid=(byte)eparms.GetValueInt(string("PhaseMkFluid")+suffix,false);
	for(int c2=0;c2<c;c2++)if(mkfluid==PhaseCte[c2].mkfluid)RunException(met,"The mk value is repeated.");
	unsigned n=parts.CountBlocks(PT_Fluid);
	unsigned idbegin,count=0;
    
	for(unsigned c2=0;c2<parts.CountBlocks();c2++){
		const JSpacePartBlock &block=parts.GetBlock(c2);
		if(block.Type==PT_Fluid){
  			const JSpacePartBlock_Fluid &fblock=(const JSpacePartBlock_Fluid &)block;
			if(fblock.GetMkType()==mkfluid){
				idbegin=fblock.GetBegin();
				count=fblock.GetCount();
			}
		}
	}

	if(!count)RunException(met,"The mk value not exist in particles.");
	//Twelve inputs for each phase
	PhaseCte[c].mkfluid=mkfluid;
	PhaseCte[c].visco=eparms.GetValueFloat(string("PhaseVisco")+suffix,false);
	PhaseCte[c].visco_cr=eparms.GetValueFloat(string("PhaseViscoCr")+suffix,true);
	PhaseCte[c].visco_max=eparms.GetValueFloat(string("PhaseViscoMax")+suffix,true);
	PhaseCte[c].coh=eparms.GetValueFloat(string("Coh")+suffix,false);
	PhaseCte[c].phi=float(PI)/180.f*eparms.GetValueFloat(string("PhiF")+suffix,false);
	PhaseCte[c].hb_n=eparms.GetValueFloat(string("HB_parameter")+suffix,false);
	PhaseCte[c].hbp_m=eparms.GetValueFloat(string("HB_Papanastasiou")+suffix,false);
	// Yield criterion and strength
	PhaseCte[c].yc=eparms.GetValueInt("YieldCriterion",true,2); 
    PhaseCte[c].ys=eparms.GetValueFloat(string("YieldStrength")+suffix,true,0.0f); 
	PhaseCte[c].sed=eparms.GetValueInt(string("Phase")+suffix,false);
    //Ids
	PhaseCte[c].idbegin=idbegin;
	PhaseCte[c].count=count;
	//calculate Morh parameters
	PhaseCte[c].DP_alpha=(tan(PhaseCte[c].phi))/(sqrt(9+12*(tan(PhaseCte[c].phi)*tan(PhaseCte[c].phi))));
	PhaseCte[c].DP_kappa=(3*PhaseCte[c].coh)/(sqrt(9+12*(tan(PhaseCte[c].phi)*tan(PhaseCte[c].phi))));
	
	PhaseArray[c].phaseid=c;
	PhaseArray[c].rho_ph=eparms.GetValueFloat(string("PhaseRhop0")+suffix,false);
	//PhaseArray[c].mass_ph=PhaseArray[c].rho_ph*Dp*Dp/**Dp*/; //if(Simulate2D)PhaseArray[c].mass_ph=PhaseArray[c].rho_ph*Dp*Dp; //**Mass has been moved to ConfigConstants**
	PhaseArray[c].Cs0_ph=eparms.GetValueFloat(string("PhaseCsound")+suffix,false); //**Cs0-max between phases has moved to ConfigConstants**
	PhaseArray[c].Gamma_ph=eparms.GetValueFloat(string("PhaseGamma")+suffix,false);
	PhaseArray[c].b_ph=PhaseArray[c].Cs0_ph*PhaseArray[c].Cs0_ph*PhaseArray[c].rho_ph/PhaseArray[c].Gamma_ph;

	char cad[1024];
	sprintf(cad,"Phase_%d->MK-fluid:%d \nID-begin:%u Count:%u \nRhop0:%f Cs0:%f Gamma:%f \n",c,mkfluid,idbegin,count,PhaseArray[c].rho_ph,PhaseArray[c].Cs0_ph,PhaseArray[c].Gamma_ph);
	Log->Print(cad);
  } 
  Phases=PhaseCount;

  Log->Print("");
}

//==============================================================================
/// Initialisation of MK information.
//==============================================================================
void JSph::ResetMkInfo(){
  delete[] MkList; MkList=NULL;
  MkListSize=MkListFixed=MkListMoving=MkListFloat=MkListBound=MkListFluid=0;
}

//==============================================================================
/// Loads MK information of particles.
//==============================================================================
void JSph::LoadMkInfo(const JSpaceParts *parts){
  ResetMkInfo();
  MkListSize=parts->CountBlocks();
  MkListFixed=parts->CountBlocks(PT_Fixed);
  MkListMoving=parts->CountBlocks(PT_Moving);
  MkListFloat=parts->CountBlocks(PT_Floating);
  MkListFluid=parts->CountBlocks(PT_Fluid);
  MkListBound=MkListFixed+MkListMoving+MkListFloat;
  //-Allocated memory.
  MkList=new StMkInfo[MkListSize];
  //-Gets info for each block of particles.
  for(unsigned c=0;c<MkListSize;c++){
    const JSpacePartBlock &block=parts->GetBlock(c);
    MkList[c].begin=block.GetBegin();
    MkList[c].count=block.GetCount();
    MkList[c].mk=block.GetMk();
    MkList[c].mktype=block.GetMkType();
    switch(block.Type){
      case PT_Fixed:     MkList[c].code=CodeSetType(0,PART_BoundFx,c);                           break;
      case PT_Moving:    MkList[c].code=CodeSetType(0,PART_BoundMv,c-MkListFixed);               break;
      case PT_Floating:  MkList[c].code=CodeSetType(0,PART_BoundFt,c-MkListFixed-MkListMoving);  break;
      case PT_Fluid:     MkList[c].code=CodeSetType(0,PART_Fluid,c-MkListBound);                 break;
    }
  }
}

//==============================================================================
/// Retunrs the block in MkList according to a given Id.
//==============================================================================
unsigned JSph::GetPosMkInfo(unsigned id)const{
  unsigned c=0;
  for(;c<MkListSize&&id>=(MkList[c].begin+MkList[c].count);c++);
  return(c);
}

//==============================================================================
/// Returns the block in MkList according to a given Id.
//==============================================================================
unsigned JSph::GetMkBlockById(unsigned id)const{
  unsigned c=0;
  for(;c<MkListSize && id>=(MkList[c].begin+MkList[c].count);c++);
  return(c);
}

//==============================================================================
/// Returns the block in MkList according to a given MK.
//==============================================================================
unsigned JSph::GetMkBlockByMk(word mk)const{
  unsigned c=0;
  for(;c<MkListSize && unsigned(mk)!=MkList[c].mk;c++);
  return(c);
}

//==============================================================================
/// Returns the code of a particle according to the given parameters.
//==============================================================================
word JSph::CodeSetType(word code,TpParticle type,unsigned value)const{
  const char met[]="CodeSetType"; 
  //-Chooses type.
  word tp; 
  if(type==PART_BoundFx)tp=CODE_TYPE_FIXED;
  else if(type==PART_BoundMv)tp=CODE_TYPE_MOVING;
  else if(type==PART_BoundFt)tp=CODE_TYPE_FLOATING;
  else if(type==PART_Fluid)tp=CODE_TYPE_FLUID;
  else RunException(met,"Type of particle is invalid.");
  //-Checks the value.
  word v=word(value&CODE_MASKVALUE);
  if(unsigned(v)!=value)RunException(met,"The value is invalid.");
  //-Returns the new code.
  return(code&(!CODE_MASKTYPEVALUE)|tp|v);
}

//==============================================================================
/// Loads the code of group of particles (moving & floating) and
/// flags the last "nout" particles as excluded.
//==============================================================================

void JSph::LoadCodeParticles(unsigned np,const unsigned *idp,word *code)const{
  const char met[]="LoadCodeParticles"; 
  //-Assigns code to each group of particles (moving & floating).
  const unsigned finfixed=CaseNfixed;
  const unsigned finmoving=finfixed+CaseNmoving;
  const unsigned finfloating=finmoving+CaseNfloat;
  for(unsigned p=0;p<np;p++){
    const unsigned id=idp[p];
    word cod=0;
    unsigned cmk=GetMkBlockById(id);
    if(id<finfixed)cod=CodeSetType(cod,PART_BoundFx,cmk);
    else if(id<finmoving){
      cod=CodeSetType(cod,PART_BoundMv,cmk-MkListFixed);
      if(cmk-MkListFixed>=MotionObjCount)RunException(met,"Motion code of particles was not found.");
    }
    else if(id<finfloating){
      cod=CodeSetType(cod,PART_BoundFt,cmk-MkListFixed-MkListMoving);
      if(cmk-MkListFixed-MkListMoving>=FtCount)RunException(met,"Floating code of particles was not found.");
    }
    else{
      cod=CodeSetType(cod,PART_Fluid,cmk-MkListBound);
      if(cmk-MkListBound>=MkListSize)RunException(met,"Fluid code of particles was not found.");
    }
    code[p]=cod;
  }
}


/* void JSph::LoadCodeParticles(unsigned np,unsigned nout,const unsigned *idp,word *code)const{
  const char met[]="LoadCodeParticles"; 
  //-Assigns code to each group of particles (moving & floating).
  const unsigned finfixed=CaseNfixed;
  const unsigned finmoving=finfixed+CaseNmoving;
  const unsigned finfloating=finmoving+CaseNfloat;
  for(unsigned p=0;p<np;p++){
    const unsigned id=idp[p];
    unsigned cmk=GetPosMkInfo(id);
    word cod=0;
    if(id<finfixed)cod=CodeSetType(cod,PART_BoundFx,cmk);
    else if(id<finmoving){
      cod=CodeSetType(cod,PART_BoundMv,cmk-MkListFixed);
      if(cmk-MkListFixed>=MotionObjCount)RunException(met,"Motion code of particles was not found.");
    }
    else if(id<finfloating){
      cod=CodeSetType(cod,PART_BoundFt,cmk-MkListFixed-MkListMoving);
      if(cmk-MkListFixed-MkListMoving>=FtCount)RunException(met,"Floating code of particles was not found.");
      //{ char cad[1024]; sprintf(cad,"++> p[%u] id:%u code:%u typevalue:%u",p,id,cod,CODE_GetTypeValue(cod)); Log->PrintDbg(cad); }
    }
    else{
      cod=CodeSetType(cod,PART_Fluid,cmk-MkListBound);
      if(cmk-MkListBound>=MkListSize)RunException(met,"Fluid code of particles was not found.");
    }
    code[p]=cod;
  }
  //-Flags excluded particles.
  for(unsigned p=np-nout;p<np;p++)code[p]=CODE_SetOutPos(code[p]);
  //RunException(met,"stop");
}
*/


//==============================================================================
/// Resizes limits of the map according to case configuration.
//==============================================================================
void JSph::ResizeMapLimits(){

  //v4.0
  Log->Print(string("MapRealPos(border)=")+fun::Double3gRangeStr(MapRealPosMin,MapRealPosMax));
  //sprintf(Cad,"MapPos(border)=(%f,%f,%f)-(%f,%f,%f)",MapPosMin.x,MapPosMin.y,MapPosMin.z,MapPosMax.x,MapPosMax.y,MapPosMax.z); Log->Print(Cad);
  tdouble3 dmin=MapRealPosMin,dmax=MapRealPosMax;
  /*
  sprintf(Cad,"\n*******************"); Log->Print(Cad);
  sprintf(Cad,"dmin.x=%f MapRealPosMin.x=%f ",dmin.x,MapRealPosMin.x); Log->Print(Cad);
  sprintf(Cad,"dmin.y=%f MapRealPosMin.y=%f ",dmin.y,MapRealPosMin.y); Log->Print(Cad);
  sprintf(Cad,"dmin.z=%f MapRealPosMin.z=%f \n",dmin.z,MapRealPosMin.z); Log->Print(Cad);
  
  sprintf(Cad,"dmax.x=%f MapRealPosMax.x=%f ",dmax.x,MapRealPosMax.x); Log->Print(Cad);
  sprintf(Cad,"dmax.y=%f MapRealPosMax.y=%f ",dmax.y,MapRealPosMax.y); Log->Print(Cad);
  sprintf(Cad,"dmax.z=%f MapRealPosMax.z=%f ",dmax.z,MapRealPosMax.z); Log->Print(Cad);  
  sprintf(Cad,"*******************\n"); Log->Print(Cad);*/

  if(CfgDomainParticles){
    tdouble3 dif=dmax-dmin;
    dmin=dmin-dif*ToTDouble3(CfgDomainParticlesPrcMin);
    dmax=dmax+dif*ToTDouble3(CfgDomainParticlesPrcMax);
    dmin=dmin-ToTDouble3(CfgDomainParticlesMin);
    dmax=dmax+ToTDouble3(CfgDomainParticlesMax);
	
 /* sprintf(Cad,"\n*******************"); Log->Print(Cad);
  sprintf(Cad,"dmin.x=%f MapRealPosMin.x=%f ",dmin.x,MapRealPosMin.x); Log->Print(Cad);
  sprintf(Cad,"dmin.y=%f MapRealPosMin.y=%f ",dmin.y,MapRealPosMin.y); Log->Print(Cad);
  sprintf(Cad,"dmin.z=%f MapRealPosMin.z=%f \n",dmin.z,MapRealPosMin.z); Log->Print(Cad);
  
  sprintf(Cad,"dmax.x=%f MapRealPosMax.x=%f ",dmax.x,MapRealPosMax.x); Log->Print(Cad);
  sprintf(Cad,"dmax.y=%f MapRealPosMax.y=%f ",dmax.y,MapRealPosMax.y); Log->Print(Cad);
  sprintf(Cad,"dmax.z=%f MapRealPosMax.z=%f ",dmax.z,MapRealPosMax.z); Log->Print(Cad);  
  sprintf(Cad,"*******************\n"); Log->Print(Cad);*/

  }
  /*else{ dmin=CfgDomainFixedMin; dmax=CfgDomainFixedMax; }
  if(dmin.x>MapPosMin.x||dmin.y>MapPosMin.y||dmin.z>MapPosMin.z||dmax.x<MapPosMax.x||dmax.y<MapPosMax.y||dmax.z<MapPosMax.z)RunException("ResizeMapLimits","Domain limits is not valid.");
  if(!PeriX){ MapPosMin.x=dmin.x; MapPosMax.x=dmax.x; }
  if(!PeriY){ MapPosMin.y=dmin.y; MapPosMax.y=dmax.y; }
  if(!PeriZ){ MapPosMin.z=dmin.z; MapPosMax.z=dmax.z; } */
  else{ dmin=ToTDouble3(CfgDomainFixedMin); dmax=ToTDouble3(CfgDomainFixedMax); }
  //sprintf(Cad,"dmin.y>MapRealPosMin.y=%d dmax.z<MapRealPosMax.z=%d ",dmin.y>MapRealPosMin.y,dmax.z<MapRealPosMax.z); Log->Print(Cad);
  //sprintf(Cad,"dmin.x>MapRealPosMin.x=%d dmin.y>MapRealPosMin.y=%d dmin.z>MapRealPosMin.z=%d dmax.x<MapRealPosMax.x=%d dmax.y<MapRealPosMax.y=%d dmax.z<MapRealPosMax.z=%d \n",dmin.x>MapRealPosMin.x, dmin.y>MapRealPosMin.y, dmin.z>MapRealPosMin.z, dmax.x<MapRealPosMax.x, dmax.y<MapRealPosMax.y, dmax.z<MapRealPosMax.z); Log->Print(Cad);
  if(dmin.x>MapRealPosMin.x||dmin.y>MapRealPosMin.y||dmin.z>MapRealPosMin.z||dmax.x<MapRealPosMax.x||dmax.y<MapRealPosMax.y||dmax.z<MapRealPosMax.z)RunException("ResizeMapLimits","Domain limits is not valid.");
  if(!PeriX){ MapRealPosMin.x=dmin.x; MapRealPosMax.x=dmax.x; }
  if(!PeriY){ MapRealPosMin.y=dmin.y; MapRealPosMax.y=dmax.y; }
  if(!PeriZ){ MapRealPosMin.z=dmin.z; MapRealPosMax.z=dmax.z; }
  //endv4
  
}

//==============================================================================
/// Configures value of constants.
//==============================================================================
void JSph::ConfigConstants(bool simulate2d){
  const char* met="ConfigConstants";
  //-Computation of constants.
  Delta2H=H*2*DeltaSph;
  
  //Cs0=sqrt(int(Gamma)*CteB/Rhop0); //-Speed of sound.
  //multi
  Cs0=0.0f;
  for(unsigned c=0;c<Phases;c++){
	  Cs0=max(Cs0,PhaseArray[c].Cs0_ph);
	  PhaseArray[c].mass_ph=PhaseArray[c].rho_ph*Dp*Dp*Dp;
	  if(Simulate2D)PhaseArray[c].mass_ph=PhaseArray[c].rho_ph*Dp*Dp;
  }


  Dosh=H*2; H2=H*H; Fourh2=H2*4; Eta2=(H*0.1f)*(H*0.1f);
  if(simulate2d){
    if(TKernel==KERNEL_Cubic){
      CubicCte.a1=float(10.0f/(PI*7.f));
      CubicCte.a2=CubicCte.a1/H2;  
      CubicCte.aa=CubicCte.a1/(H*H*H);       
      CubicCte.a24=0.25f*CubicCte.a2;
      CubicCte.c1=-3.0f*CubicCte.aa;
      CubicCte.d1=9.0f*CubicCte.aa/4.0f;
      CubicCte.c2=-3.0f*CubicCte.aa/4.0f;
      float deltap=1.f/1.5f;
      float wdeltap=CubicCte.a2*(1.f-1.5f*deltap*deltap+0.75f*deltap*deltap*deltap);
      CubicCte.od_wdeltap=1.f/wdeltap;
      CteShepard=CubicCte.a2;  

    }
    if(TKernel==KERNEL_Wendland){
      WendlandCte.awen=0.557f/(H*H);
      WendlandCte.bwen=-2.7852f/(H*H*H);
      CteShepard=WendlandCte.awen;  
    }
  }
  else{
    if(TKernel==KERNEL_Cubic){
      CubicCte.a1=float(1.0f/PI);
      CubicCte.a2=CubicCte.a1/(H*H*H); 
      CubicCte.aa=CubicCte.a1/(H*H*H*H);       
      CubicCte.a24=0.25f*CubicCte.a2;
      CubicCte.c1=-3.0f*CubicCte.aa;
      CubicCte.d1=9.0f*CubicCte.aa/4.0f;
      CubicCte.c2=-3.0f*CubicCte.aa/4.0f;
      float deltap=1.f/1.5f;
      float wdeltap=CubicCte.a2*(1.f-1.5f*deltap*deltap+0.75f*deltap*deltap*deltap);
      CubicCte.od_wdeltap=1.f/wdeltap;
      CteShepard=CubicCte.a2;  
    }
    if(TKernel==KERNEL_Wendland){
      WendlandCte.awen=0.41778f/(H*H*H);
      WendlandCte.bwen=-2.08891f/(H*H*H*H);
      CteShepard=WendlandCte.awen; 
    }
  }
  //-Constants for Laminar viscosity + SPS turbulence model.
  if(TVisco==VISCO_LaminarSPS || TVisco==VISCO_SumSPS){  
    float dp_sps=(Simulate2D? sqrt(2.0f*Dp*Dp)/2.0f: sqrt(3.0f*Dp*Dp)/3.0f);  
    SpsSmag=pow((0.12f*dp_sps),2);
    SpsBlin=(2.0f/3.0f)*0.0066f*dp_sps*dp_sps; 
  }



  VisuConfig(); 
}

//==============================================================================
/// Prints out configuration of the case.
//==============================================================================
void JSph::VisuConfig()const{
  const char* met="VisuConfig";
  Log->Print(Simulate2D? "**2D-Simulation parameters:": "**3D-Simulation parameters:");
  Log->Print(fun::VarStr("CaseName",CaseName));
  Log->Print(fun::VarStr("RunName",RunName));
  Log->Print(fun::VarStr("SvTimers",SvTimers));
  Log->Print(fun::VarStr("StepAlgorithm",GetStepName(TStep)));
  if(TStep==STEP_None)RunException(met,"StepAlgorithm value is invalid.");
  if(TStep==STEP_Verlet)Log->Print(fun::VarStr("VerletSteps",VerletSteps));
  Log->Print(fun::VarStr("Kernel",GetKernelName(TKernel)));
  Log->Print(fun::VarStr("Viscosity",GetViscoName(TVisco)));
  //Log->Print(fun::VarStr("Visco",Visco));
  Log->Print(fun::VarStr("ShepardSteps",ShepardSteps));
  Log->Print(fun::VarStr("DeltaSph",GetDeltaSphName(TDeltaSph)));
  if(TDeltaSph!=DELTA_None)Log->Print(fun::VarStr("DeltaSphValue",DeltaSph));
  Log->Print(fun::VarStr("CaseNp",CaseNp));
  Log->Print(fun::VarStr("CaseNbound",CaseNbound));
  Log->Print(fun::VarStr("CaseNfixed",CaseNfixed));
  Log->Print(fun::VarStr("CaseNmoving",CaseNmoving));
  Log->Print(fun::VarStr("CaseNfloat",CaseNfloat));
  Log->Print(fun::VarStr("PeriodicActive",PeriActive));
  if(PeriXY)Log->Print(fun::VarStr("PeriodicXY",PeriXY));
  if(PeriXZ)Log->Print(fun::VarStr("PeriodicXZ",PeriXZ));
  if(PeriYZ)Log->Print(fun::VarStr("PeriodicYZ",PeriYZ));
  if(PeriX)Log->Print(fun::VarStr("PeriodicXinc",PeriXinc));
  if(PeriY)Log->Print(fun::VarStr("PeriodicYinc",PeriYinc));
  if(PeriZ)Log->Print(fun::VarStr("PeriodicZinc",PeriZinc));
  Log->Print(fun::VarStr("Dx",Dp));
  Log->Print(fun::VarStr("H",H));
  Log->Print(fun::VarStr("CteB",CteB));
  Log->Print(fun::VarStr("Gamma",Gamma));
  //Log->Print(fun::VarStr("Rhop0",Rhop0)); 
  Log->Print(fun::VarStr("Eps",Eps));
  Log->Print(fun::VarStr("Cs0 (max)",Cs0));
  Log->Print(fun::VarStr("CFLnumber",CFLnumber));
  Log->Print(fun::VarStr("DtIni",DtIni));
  if(DtFixed)Log->Print(fun::VarStr("DtFixed",DtFixed->GetFile()));
  Log->Print(fun::VarStr("DtMin",DtMin));
  Log->Print(fun::VarStr("MassFluid",MassFluid));
  Log->Print(fun::VarStr("MassBound",MassBound));
  if(TKernel==KERNEL_Cubic){
    Log->Print(fun::VarStr("CubicCte.a1",CubicCte.a1));
    Log->Print(fun::VarStr("CubicCte.aa",CubicCte.aa));
    Log->Print(fun::VarStr("CubicCte.a24",CubicCte.a24));
    Log->Print(fun::VarStr("CubicCte.c1",CubicCte.c1));
    Log->Print(fun::VarStr("CubicCte.c2",CubicCte.c2));
    Log->Print(fun::VarStr("CubicCte.d1",CubicCte.d1));
    Log->Print(fun::VarStr("CubicCte.od_wdeltap",CubicCte.od_wdeltap));
  }
  else if(TKernel==KERNEL_Wendland){
    Log->Print(fun::VarStr("WendlandCte.awen",WendlandCte.awen));
    Log->Print(fun::VarStr("WendlandCte.bwen",WendlandCte.bwen));
  }
  if(TVisco==VISCO_LaminarSPS){     
    Log->Print(fun::VarStr("SpsSmag",SpsSmag));
    Log->Print(fun::VarStr("SpsBlin",SpsBlin));
  } 
  if(ShepardSteps)Log->Print(fun::VarStr("CteShepard",CteShepard)); 
  Log->Print(fun::VarStr("TimeMax",TimeMax));
  Log->Print(fun::VarStr("TimePart",TimePart));
  Log->Print(fun::VarStr("Gravity",Gravity));
  Log->Print(fun::VarStr("PartOutMax",(int)PartOutMax));
  Log->Print(fun::VarStr("RhopOut",RhopOut));
  if(RhopOut){
    Log->Print(fun::VarStr("RhopOutMin",RhopOutMin));
    Log->Print(fun::VarStr("RhopOutMax",RhopOutMax));
  }
  if(VarAcc)Log->Print(fun::VarStr("VarAcc",VarAcc->GetBaseFile()+":"+fun::UintStr(VarAcc->GetCount())));
  if(CaseNfloat)Log->Print(fun::VarStr("FtPause",FtPause));
  if(CteB==0)RunException(met,"Constant \'b\' can not be zero.\n\'b\' is zero when fluid height is zero (or fluid particles were not created)");
}

//==============================================================================
/// Configures CellOrder and adjusts order of components in data.
//==============================================================================
void JSph::ConfigCellOrder(TpCellOrder order,unsigned np,tfloat3* pos,tfloat3* vel){
  CellOrder=order;
  if(CellOrder==ORDER_None)CellOrder=ORDER_XYZ;
  if(Simulate2D&&CellOrder!=ORDER_XYZ&&CellOrder!=ORDER_ZYX)RunException("ConfigCellOrder","In 2D simulations the value of CellOrder must be XYZ or ZYX.");
  Log->Print(fun::VarStr("CellOrder",string(GetNameCellOrder(CellOrder))));
  if(CellOrder!=ORDER_XYZ){
    //-Modifies initial data of particles.
    OrderCodeData(CellOrder,np,pos);
    OrderCodeData(CellOrder,np,vel);
    //-Modifies other constants.
    Gravity=OrderCodeValue(CellOrder,Gravity);
    MapPosMin=OrderCodeValue(CellOrder,MapPosMin);
    MapPosMax=OrderCodeValue(CellOrder,MapPosMax);
    //-Modifies variables of floating bodies.
    for(unsigned cf=0;cf<FtCount;cf++){
      StFloatingData *fobj=FtObjs+cf;
      fobj->inertia=OrderCodeValue(CellOrder,fobj->inertia);
      fobj->center=OrderCodeValue(CellOrder,fobj->center);
      fobj->fvel=OrderCodeValue(CellOrder,fobj->fvel);
      fobj->fomega=OrderCodeValue(CellOrder,fobj->fomega);
    }
    //-Modifies configuration of periodic boundaries.
    bool perix=PeriX,periy=PeriY,periz=PeriZ;
    bool perixy=PeriXY,perixz=PeriXZ,periyz=PeriYZ;
    tfloat3 perixinc=PeriXinc,periyinc=PeriYinc,perizinc=PeriZinc;
    //sprintf(Cad,"CellOrder---->0 PeriX:%s PeriY:%s PeriZ:%s",(PeriX?"True":"False"),(PeriY?"True":"False"),(PeriZ?"True":"False")); Log->PrintDbg(Cad);
    //sprintf(Cad,"CellOrder---->0 PeriXinc:%s PeriYinc:%s PeriZinc:%s",fun::Float3gStr(PeriXinc).c_str(),fun::Float3gStr(PeriYinc).c_str(),fun::Float3gStr(PeriZinc).c_str()); Log->PrintDbg(Cad);
    tuint3 v={1,2,3};
    //sprintf(Cad,"CellOrder---->0 v:(%u,%u,%u)",v.x,v.y,v.z); Log->PrintDbg(Cad);
    v=OrderCode(v);
    //sprintf(Cad,"CellOrder---->1 v:(%u,%u,%u)",v.x,v.y,v.z); Log->PrintDbg(Cad);
    if(v.x==2){ PeriX=periy; PeriXinc=OrderCode(periyinc); }
    if(v.x==3){ PeriX=periz; PeriXinc=OrderCode(perizinc); }
    if(v.y==1){ PeriY=perix; PeriYinc=OrderCode(perixinc); }
    if(v.y==3){ PeriY=periz; PeriYinc=OrderCode(perizinc); }
    if(v.z==1){ PeriZ=perix; PeriZinc=OrderCode(perixinc); }
    if(v.z==2){ PeriZ=periy; PeriZinc=OrderCode(periyinc); }
    if(perixy){
      PeriXY=(CellOrder==ORDER_XYZ||CellOrder==ORDER_YXZ);
      PeriXZ=(CellOrder==ORDER_XZY||CellOrder==ORDER_YZX);
      PeriYZ=(CellOrder==ORDER_ZXY||CellOrder==ORDER_ZYX);
    }
    if(perixz){
      PeriXY=(CellOrder==ORDER_XZY||CellOrder==ORDER_ZXY);
      PeriXZ=(CellOrder==ORDER_XYZ||CellOrder==ORDER_ZYX);
      PeriYZ=(CellOrder==ORDER_YXZ||CellOrder==ORDER_YZX);
    }
    if(periyz){
      PeriXY=(CellOrder==ORDER_YZX||CellOrder==ORDER_ZYX);
      PeriXZ=(CellOrder==ORDER_YXZ||CellOrder==ORDER_ZXY);
      PeriYZ=(CellOrder==ORDER_XYZ||CellOrder==ORDER_XZY);
    }
    //sprintf(Cad,"CellOrder---->1 PeriX:%s PeriY:%s PeriZ:%s",(PeriX?"True":"False"),(PeriY?"True":"False"),(PeriZ?"True":"False")); Log->PrintDbg(Cad);
    //sprintf(Cad,"CellOrder---->1 PeriXinc:%s PeriYinc:%s PeriZinc:%s",fun::Float3gStr(PeriXinc).c_str(),fun::Float3gStr(PeriYinc).c_str(),fun::Float3gStr(PeriZinc).c_str()); Log->PrintDbg(Cad);
  }
  PeriActive=(PeriX? 1: 0)+(PeriY? 2: 0)+(PeriZ? 4: 0);
}

//==============================================================================
/// Modifies order of components of an array of type tfloat3.
//==============================================================================
void JSph::OrderCodeData(TpCellOrder order,unsigned n,tfloat3 *v){
  if(order==ORDER_XZY)for(unsigned c=0;c<n;c++)v[c]=ReOrderXZY(v[c]);
  if(order==ORDER_YXZ)for(unsigned c=0;c<n;c++)v[c]=ReOrderYXZ(v[c]);
  if(order==ORDER_YZX)for(unsigned c=0;c<n;c++)v[c]=ReOrderYZX(v[c]);
  if(order==ORDER_ZXY)for(unsigned c=0;c<n;c++)v[c]=ReOrderZXY(v[c]);
  if(order==ORDER_ZYX)for(unsigned c=0;c<n;c++)v[c]=ReOrderZYX(v[c]);
}

//==============================================================================
/// Shows a message with the allocated memory for main data of particles.
//==============================================================================
void JSph::PrintSizeNp(unsigned np,long long size)const{
  char cad[128];
  sprintf(cad,"**Requested %s memory for %u particles: %.1f MB.",(Cpu? "cpu": "gpu"),np,double(size)/(1024*1024)); 
  Log->Print(cad);
}

//==============================================================================
/// Displays headers of PARTs
//==============================================================================
void JSph::PrintHeadPart(){
  if(SvSteps){
    Log->Print("PART-Step           PartTime      Time/Seg   Finish time        ");
    Log->Print("==================  ============  =========  ===================");
  }
  else{
    Log->Print("PART       PartTime      TotalSteps    Steps    Time/Seg   Finish time        ");
    Log->Print("=========  ============  ============  =======  =========  ===================");
  }
  fflush(stdout);
}

//==============================================================================
/// Prepares the recording of data of floating objects and
/// loads previous data in case of "restart".
//==============================================================================
void JSph::InitFloatingData(){
  FtData=new JFloatingData();
  FtData->Config(Dp,H,CteB,Rhop0,Gamma,Simulate2D,CaseNp,CaseNfixed,CaseNmoving,CaseNfloat,FtCount); 
  if(PartBegin){//-Loads data of a previous simulation.
    FtData->LoadFile(fun::GetDirWithSlash(PartBeginDir)+FtFileData);
    const JFloatingDataPart* fpart=FtData->GetByPart(PartBegin);
    if(!fpart)RunException("InitFloatingData","There is not initial data of floating body to restart.");
    for(unsigned cf=0;cf<FtCount;cf++){
      JFloatingDataPart::StFloatingData fdat=fpart->GetFtData(cf);
      StFloatingData *fo=FtObjs+cf;
      fo->begin=fdat.begin;
      fo->count=fdat.count;
      fo->mass=fdat.mass;
      fo->massp=fdat.massp;
      fo->inertia=fdat.inertia;
      fo->center=fdat.center;
      fo->fvel=fdat.fvel;
      fo->fomega=fdat.fomega;
    }
    FtData->RemoveParts(PartBeginFirst);
  }
}

//==============================================================================
/// Generates output file of floating bodies.
//==============================================================================
void JSph::SaveFloatingData(){
  //-Adds data of floatings.
  JFloatingDataPart* fpart=FtData->AddPart(Part,TimeStep);
  for(unsigned cf=0;cf<FtCount;cf++){
    StFloatingData *fo=FtObjs+cf;
    fpart->AddFloating(fo->begin,fo->count,fo->mass,fo->massp,OrderDecodeValue(CellOrder,fo->inertia),OrderDecodeValue(CellOrder,fo->center),OrderDecodeValue(CellOrder,fo->fvel),OrderDecodeValue(CellOrder,fo->fomega));
  }
  FtData->SaveFile(DirOut+FtFileData,FtData->GetDist(CaseNfloat)==NULL);
  FtData->ResetData();
}

//==============================================================================
/// Generates CSV output file with all info of floating bodies at the end.
//==============================================================================
void JSph::SaveFloatingDataTotal(){
  FtData->LoadFile(DirOut+FtFileData);
  FtData->SaveFileCsv(DirOut+fun::GetWithoutExtension(FtFileData)+".csv");
}


//==============================================================================
// Establece configuracion para grabacion de particulas.
//==============================================================================
void JSph::ConfigSaveData(unsigned piece,unsigned pieces,std::string div){
  const char met[]="ConfigSaveData";
  //-Configura objeto para grabacion de particulas e informacion.
  if(SvData&SDAT_Info || SvData&SDAT_Binx){
    DataBi4=new JPartDataBi4();
    DataBi4->ConfigBasic(piece,pieces,RunCode,AppName,Simulate2D,DirOut);
    DataBi4->ConfigParticles(CaseNp,CaseNfixed,CaseNmoving,CaseNfloat,CaseNfluid,CasePosMin,CasePosMax,NpDynamic,ReuseIds);
    DataBi4->ConfigCtes(Dp,H,CteB,Rhop0,Gamma,MassBound,MassFluid);
    DataBi4->ConfigSimMap(OrderDecode(MapRealPosMin),OrderDecode(MapRealPosMax));
    JPartDataBi4::TpPeri tperi=JPartDataBi4::PERI_None;
    //if(0){//PeriodicConfig.PeriActive){
    //  if(PeriodicConfig.PeriXY)tperi=JPartDataBi4::PERI_XY;
    //  else if(PeriodicConfig.PeriXZ)tperi=JPartDataBi4::PERI_XZ;
    //  else if(PeriodicConfig.PeriYZ)tperi=JPartDataBi4::PERI_YZ;
    //  else if(PeriodicConfig.PeriX)tperi=JPartDataBi4::PERI_X;
    //  else if(PeriodicConfig.PeriY)tperi=JPartDataBi4::PERI_Y;
    //  else if(PeriodicConfig.PeriZ)tperi=JPartDataBi4::PERI_Z;
    //  else RunException(met,"The periodic configuration is invalid.");
    //}
    //DataBi4->ConfigSimPeri(tperi,PeriodicConfig.PeriXinc,PeriodicConfig.PeriYinc,PeriodicConfig.PeriZinc);
    //if(div.empty())
		DataBi4->ConfigSimDiv(JPartDataBi4::DIV_None);
    //else if(div=="X")DataBi4->ConfigSimDiv(JPartDataBi4::DIV_X);
    //else if(div=="Y")DataBi4->ConfigSimDiv(JPartDataBi4::DIV_Y);
    //else if(div=="Z")DataBi4->ConfigSimDiv(JPartDataBi4::DIV_Z);
    //else RunException(met,"The division configuration is invalid.");
  }
  //-Configura objeto para grabacion de particulas excluidas.
  if(SvData&SDAT_Binx){
    DataOutBi4=new JPartOutBi4Save();
    DataOutBi4->ConfigBasic(piece,pieces,RunCode,AppName,Simulate2D,DirOut);
    DataOutBi4->ConfigParticles(CaseNp,CaseNfixed,CaseNmoving,CaseNfloat,CaseNfluid);
    DataOutBi4->ConfigLimits(OrderDecode(MapRealPosMin),OrderDecode(MapRealPosMax),(RhopOut? RhopOutMin: 0),(RhopOut? RhopOutMax: 0));
    DataOutBi4->SaveInitial();
  }
  //-Configura objeto para grabacion de datos de floatings.
  /*if(SvData&SDAT_Binx && FtCount){
    DataFloatBi4=new JPartFloatBi4Save();
    DataFloatBi4->Config(AppName,DirOut,FtCount);
    for(unsigned cf=0;cf<FtCount;cf++)DataFloatBi4->AddHeadData(cf,FtObjs[cf].mkbound,FtObjs[cf].begin,FtObjs[cf].count,FtObjs[cf].mass,FtObjs[cf].radius);
    DataFloatBi4->SaveInitial();
  }*/
  //-Crea objeto para almacenar las particulas excluidas hasta su grabacion.
  PartsOut=new JPartsOut();
}


//==============================================================================
// Almacena nuevas particulas excluidas hasta la grabacion del proximo PART.
//==============================================================================
void JSph::AddParticlesOut(unsigned nout,const unsigned *idp,const tfloat3* pos,const tfloat3 *vel,const float *rhop,unsigned noutrhop,unsigned noutmove){
  PartsOut->AddParticles(nout,idp,pos,vel,rhop,noutrhop,noutmove);
}

//v4
//==============================================================================
// Graba los ficheros de datos de particulas.
//==============================================================================
void JSph::SavePartData(unsigned npok,unsigned nout,const unsigned *idp,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float*viscop,const tsymatrix3f *vtau,const unsigned *idpm ,unsigned ndom,const tdouble3 *vdom,const StInfoPartPlus *infoplus){
  //-Graba datos de particulas y/o informacion en formato bi4.

	// I know the problem is here, if(DataBi4){ .. } I dont know why when binx it executes ok but when vtk if(DataBi4){ .. } should be false,
	// but I dont know why it is not ! if I comment the if(DataBi4){ .. } manually it works fine with vtk 
  if(DataBi4){
    tfloat3* posf3=NULL;
    TimerPart.Stop();
    JBinaryData* bdat=DataBi4->AddPartInfo(Part,TimeStep,npok,nout,Nstep,TimerPart.GetElapsedTimeD()/1000.,vdom[0],vdom[1],TotalNp);
    if(infoplus && SvData&SDAT_Info){
      bdat->SetvDouble("dtmean",(!Nstep? 0: (TimeStep-TimeStepM1)/(Nstep-PartNstep)));
      bdat->SetvDouble("dtmin",(!Nstep? 0: PartDtMin));
      bdat->SetvDouble("dtmax",(!Nstep? 0: PartDtMax));
      if(DtFixed)bdat->SetvDouble("dterror",DtFixed->GetDtError(true));
      bdat->SetvDouble("timesim",infoplus->timesim);
      bdat->SetvUint("nct",infoplus->nct);
      bdat->SetvUint("npbin",infoplus->npbin);
      bdat->SetvUint("npbout",infoplus->npbout);
      bdat->SetvUint("npf",infoplus->npf);
      bdat->SetvUint("npbper",infoplus->npbper);
      bdat->SetvUint("npfper",infoplus->npfper);
      bdat->SetvLlong("cpualloc",infoplus->memorycpualloc);
      if(infoplus->gpudata){
        bdat->SetvLlong("nctalloc",infoplus->memorynctalloc);
        bdat->SetvLlong("nctused",infoplus->memorynctused);
        bdat->SetvLlong("npalloc",infoplus->memorynpalloc);
        bdat->SetvLlong("npused",infoplus->memorynpused);
      }
    }
    if(SvData&SDAT_Binx){
      //if(SvDouble)
		DataBi4->AddPartData(npok,idp,pos,vel,rhop);
      //else{
        //posf3=GetPointerDataFloat3(npok,pos);
        //->AddPartData(npok,idp,posf3,vel,rhop);
      //}
      DataBi4->SaveFilePart();
    }
    if(SvData&SDAT_Info)DataBi4->SaveFileInfo();
    delete[] posf3;
  }

  //-Graba ficheros VKT y/o CSV.
  if((SvData&SDAT_Csv)||(SvData&SDAT_Vtk)){
    //-Genera array con posf3 y tipo de particula.
    //tfloat3* posf3=GetPointerDataFloat3(npok,pos);
    byte *type=new byte[npok];
    for(unsigned p=0;p<npok;p++){
      const unsigned id=idp[p];
      type[p]=(id>=CaseNbound? 3: (id<CaseNfixed? 0: (id<CaseNpb? 1: 2)));
    }
    //-Define campos a grabar.
    JFormatFiles2::StScalarData fields[8];
    unsigned nfields=0;
    if(idp){   fields[nfields]=JFormatFiles2::DefineField("Id",JFormatFiles2::UInt32,1,idp);		nfields++; }
    if(vel){   fields[nfields]=JFormatFiles2::DefineField("Vel",JFormatFiles2::Float32,3,vel);		nfields++; }
    if(rhop){  fields[nfields]=JFormatFiles2::DefineField("Rhop",JFormatFiles2::Float32,1,rhop);	nfields++; }
	//extra
	if(press){  fields[nfields]=JFormatFiles2::DefineField("Press",JFormatFiles2::Float32,1,press); nfields++; }
	if(viscop){  fields[nfields]=JFormatFiles2::DefineField("Viscop",JFormatFiles2::Float32,1,viscop);  nfields++; }
	if(idpm){  fields[nfields]=JFormatFiles2::DefineField("Id-phase",JFormatFiles2::UInt32,1,idpm);  nfields++; }
    if(type){  fields[nfields]=JFormatFiles2::DefineField("Type",JFormatFiles2::UChar8,1,type);		nfields++; }

    if(SvData&SDAT_Vtk)JFormatFiles2::SaveVtk(DirOut+fun::FileNameSec("PartVtk.vtk",Part),npok,pos,nfields,fields);
    if(SvData&SDAT_Csv)JFormatFiles2::SaveCsv(DirOut+fun::FileNameSec("PartCsv.csv",Part),npok,pos,nfields,fields);
    //-libera memoria.
    //delete[] posf3;
    delete[] type; 
  }

  //-Graba datos de particulas excluidas.
  if(DataOutBi4 && PartsOut->GetCount()){
    //if(SvDouble)
		DataOutBi4->SavePartOut(Part,TimeStep,PartsOut->GetCount(),PartsOut->GetIdpOut(),PartsOut->GetPosOut(),PartsOut->GetVelOut(),PartsOut->GetRhopOut());
    //else{
      //const tfloat3* posf3=GetPointerDataFloat3(PartsOut->GetCount(),PartsOut->GetPosOut());
      //DataOutBi4->SavePartOut(Part,TimeStep,PartsOut->GetCount(),PartsOut->GetIdpOut(),posf3,PartsOut->GetVelOut(),PartsOut->GetRhopOut());
      //delete[] posf3;
    //}
  }

  //-Graba datos de floatings.
  if(DataFloatBi4){
    //if(CellOrder==ORDER_XYZ)for(unsigned cf=0;cf<FtCount;cf++)DataFloatBi4->AddPartData(cf,FtObjs[cf].center,FtObjs[cf].fvel,FtObjs[cf].fomega);
    //else                    for(unsigned cf=0;cf<FtCount;cf++)DataFloatBi4->AddPartData(cf,OrderDecodeValue(CellOrder,FtObjs[cf].center),OrderDecodeValue(CellOrder,FtObjs[cf].fvel),OrderDecodeValue(CellOrder,FtObjs[cf].fomega));
    //DataFloatBi4->SavePartFloat(Part,TimeStep,DemDtForce);
  }

  //-Vacia almacen de particulas excluidas.
  PartsOut->Clear();
}

/*
//v3
//==============================================================================
/// Generates output files starting from BinData.
//==============================================================================
void JSph::SaveBinData(const char *suffixpart,unsigned npok,unsigned nout,const unsigned *idp,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float*viscop,const tsymatrix3f *vtau,const unsigned *idpm){
  const char met[]="SaveBinData";
  //-Generates BinData if no exists.
  if(!BinData){
    BinData=new JPartData();
    BinData->Config(JPartData::FmtBi2,CaseNp,CaseNbound,CaseNfluid,CaseNfixed,CaseNmoving,CaseNfloat,Dp,H,CteB,Rhop0,Gamma,MassBound,MassFluid,Simulate2D);
  }
  //-Adds data to BinData.
  if(BinData->SetDataUnsorted(Part,TimeStep,false,npok,nout,idp,pos,vel,rhop,press,viscop,vtau,idpm))RunException(met,"Some excluded particles appear again in the simulation.");
  if(SvSteps){
    BinData->SetMaskFileName(JPartData::FmtAscii,"PART_%08d");
    BinData->SetMaskFileName(JPartData::FmtBin,"PartBinx_%08d.bin");
    BinData->SetMaskFileName(JPartData::FmtBi2,"Part%08d.bi2");
  }
  //-Output formats.
  //if(SvData&SDAT_Sphysics)BinData->SaveFile(JPartData::FmtAscii,DirOut);
  
  if(SvData&SDAT_Binx){
    BinData->SaveFile(JPartData::FmtBi2,DirOut);
    BinData->SaveFile(JPartData::FmtBi2Out,DirOut,Part!=PartIni);
	
	printf("step 1");
	DataBi4->AddPartData(npok,idp,pos,vel,rhop);

	DataBi4->SaveFilePart();
	printf("step 3");
  } 
  
  if((SvData&SDAT_Csv)||(SvData&SDAT_Vtk)){
    tfloat3 *pos,*vel;
    float *rhop;
    unsigned *id;
	//multi
	float *press,*viscop;
	tsymatrix3f *vtau;
	unsigned *idpm;

    BinData->GetDataPointers(id,pos,vel,rhop,press,viscop,vtau,idpm);
    BinData->SortDataOut();
    if(SvData&SDAT_Csv)JFormatFiles2::ParticlesToCsv(DirOut+"PartCsv"+suffixpart+".csv",BinData->GetNp(),BinData->GetNfixed(),BinData->GetNmoving(),BinData->GetNfloat(),BinData->GetNfluid()-BinData->GetNfluidOut(),BinData->GetNfluidOut(),BinData->GetPartTime(),pos,vel,rhop,NULL,NULL,id,NULL,NULL,NULL,NULL);
    //if(SvData&SDAT_Vtk)JFormatFiles2::ParticlesToVtk(DirOut+"PartVtk"+suffixpart+".vtk",BinData->GetNp(),pos,vel,rhop,NULL,NULL,id,NULL,NULL,NULL,NULL);
	if(SvData&SDAT_Vtk)JFormatFiles2::SaveVtkBasic(DirOut+"PartVtk"+suffixpart+".vtk",BinData->GetNp(),id,pos,vel,rhop,press,viscop,vtau,idpm);


  }
}
*/

//v4
//==============================================================================
// Genera los ficheros de salida de datos
//==============================================================================
void JSph::SaveData(unsigned npok,const unsigned *idp,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float*viscop,const tsymatrix3f *vtau,const unsigned *idpm,unsigned ndom,const tdouble3 *vdom,const StInfoPartPlus *infoplus)
{
  const char met[]="SaveData";
  string suffixpartx=fun::PrintStr("_%04d",Part);
  //printf("1\n");getchar();
  //-Contabiliza nuevas particulas excluidas
  const unsigned noutpos=PartsOut->GetOutPosCount(),noutrhop=PartsOut->GetOutRhopCount(),noutmove=PartsOut->GetOutMoveCount();
  //printf("2\n");getchar();
  const unsigned nout=noutpos+noutrhop+noutmove;
  AddOutCount(noutpos,noutrhop,noutmove);
  //printf("3\n");getchar();

  //-Graba ficheros con datos de particulas.
  SavePartData(npok,nout,idp,pos,vel,rhop,press,viscop,vtau,idpm,ndom,vdom,infoplus);
  
  //-Reinicia limites de dt
  PartDtMin=FLT_MAX; PartDtMax=-FLT_MAX;

  //-Calculo de tiempo
  if(Part>PartIni||Nstep){
    TimerPart.Stop();
    double tpart=TimerPart.GetElapsedTimeD()/1000;
    double tseg=tpart/(TimeStep-TimeStepM1);
    TimerSim.Stop();
    double tcalc=TimerSim.GetElapsedTimeD()/1000;
    double tleft=(tcalc/(TimeStep-TimeStepIni))*(TimeMax-TimeStep);
    Log->Printf("Part%s  %12.6f  %12d  %7d  %9.2f  %14s",suffixpartx.c_str(),TimeStep,(Nstep+1),Nstep-PartNstep,tseg,fun::GetDateTimeAfter(int(tleft)).c_str());
  }
  else Log->Printf("Part%s        %u particles successfully stored",suffixpartx.c_str(),npok);   
  
  //-Muestra info de particulas excluidas.
  if(nout){
    PartOut+=nout;
    Log->Printf("  Particles out: %u  (total: %u)",nout,PartOut);
  }

  if(SvDomainVtk)SaveDomainVtk(ndom,vdom);
}

/*
//v3
//==============================================================================
/// Generates output files.
//==============================================================================
void JSph::SaveData(unsigned npok,unsigned noutpos,unsigned noutrhop,unsigned noutmove,const unsigned *idp,const tfloat3 *pos,const tfloat3 *vel,const float *rhop,const float *press,const float*viscop,const tsymatrix3f *vtau,const unsigned *idpm,unsigned ndom,const tfloat3 *vdom){
  const char met[]="SaveData";
  char suffixpart[64];
  if(SvSteps){
    if(Part)sprintf(suffixpart,"_%08d",Nstep); else sprintf(suffixpart,"_--------");
  }
  else sprintf(suffixpart,"_%04d",Part);

  //-Counts new excluded particles.
  AddOutCount(noutpos,noutrhop,noutmove);
  const unsigned nout=noutpos+noutrhop+noutmove;

  //-Stores files of particles.
  if(SvData!=SDAT_None)SaveBinData(suffixpart,npok,nout,idp,pos,vel,rhop,press,viscop,vtau,idpm);

  //-Stores data of floating objects.
  if(CaseNfloat)SaveFloatingData();

  //-Restarts limits of dt.
  PartDtMin=FLT_MAX; PartDtMax=-FLT_MAX;

  //-Computes time.
  char cad[128];
  if(Part>PartIni||Nstep){
    TimerPart.Stop();
    float tpart=TimerPart.GetElapsedTimeF()/1000.f;
    float tseg=tpart/(TimeStep-TimeStepM1);
    TimerSim.Stop();
    float tcalc=TimerSim.GetElapsedTimeF()/1000.f;
    float tleft=(tcalc/(TimeStep-TimeStepIni))*(TimeMax-TimeStep);
    if(SvSteps)sprintf(cad,"Part%s  %12.6f  %9.2f  %14s",suffixpart,TimeStep,tseg,fun::GetDateTimeAfter(int(tleft)).c_str());
    else sprintf(cad,"Part%s  %12.6f  %12d  %7d  %9.2f  %14s",suffixpart,TimeStep,(Nstep+1),Nstep-PartNstep,tseg,fun::GetDateTimeAfter(int(tleft)).c_str());
    //sprintf(cad,"Part%s  %18.12f  %12d  %7d  <-Dbg",suffixpart,TimeStep,(Nstep+1),Nstep-PartNstep); 
    Log->Print(cad);
  }
  else{
    sprintf(cad,"Part%s        %u particles successfully stored",suffixpart,npok+nout);   
    Log->Print(cad);
  }
  if(nout){
    PartOut+=nout;
    sprintf(cad,"Particles out: %u  (total: %u)",nout,PartOut);
    Log->Print(cad);
  }
  if(SvDomainVtk)SaveDomainVtk(ndom,vdom);
}

*/

////==============================================================================
///// Generates VTK file with the domain of the particles.
////==============================================================================
//void JSph::SaveDomainVtk(unsigned ndom,const tfloat3 *vdom)const{ 
//  if(vdom){
//    string fname="Domain";
//    char cad[256];
//    if(SvSteps)sprintf(cad,"_%08d",Nstep);
//    else sprintf(cad,"_%04d",Part);
//    fname=fname+cad+".vtk";
//    JFormatFiles2::SaveVtkBoxes(DirOut+fname,ndom,vdom,H*0.5f);
//  }
//}

//v4
//==============================================================================
// Genera fichero VTK con el dominio de las particulas.
//==============================================================================
void JSph::SaveDomainVtk(unsigned ndom,const tdouble3 *vdom)const{ 
  if(vdom){
    string fname=fun::FileNameSec("Domain.vtk",Part);
    tfloat3 *vdomf3=new tfloat3[ndom*2];
    for(unsigned c=0;c<ndom*2;c++)vdomf3[c]=ToTFloat3(vdom[c]);
    JFormatFiles2::SaveVtkBoxes(DirOut+fname,ndom,vdomf3,H*0.5f);
    delete[] vdomf3;
  }
}

//==============================================================================
/// Generates VTK file with the cells of the map.
//==============================================================================
void JSph::SaveMapCellsVtk(float scell)const{
  JFormatFiles2::SaveVtkCells(DirOut+"MapCells.vtk",OrderDecode(MapPosMin),OrderDecode(MapCells),scell);
}

//==============================================================================
/// Adds the basic information of summary to hinfo & dinfo. 
//==============================================================================
void JSph::GetResInfo(float tsim,float ttot,const std::string &headplus,const std::string &detplus,std::string &hinfo,std::string &dinfo){
  hinfo=hinfo+"#RunName;RunCode;DateTime;Np;TSimul;TSeg;TTotal;MemCpu;MemGpu;Steps;PartFiles;PartsOut;MaxParticles;MaxCells;Hw;StepAlgo;Kernel;Viscosity;ViscoValue;DeltaSPH;Shepard;TMax;Nbound;Nfixed;H;RhopOut;PartsRhopOut;PartsVelOut;CellMode"+headplus;
  dinfo=dinfo+ RunName+ ";"+ RunCode+ ";"+ RunTimeDate+ ";"+ fun::UintStr(CaseNp);
  dinfo=dinfo+ ";"+ fun::FloatStr(tsim)+ ";"+ fun::FloatStr(tsim/TimeStep)+ ";"+ fun::FloatStr(ttot);
  dinfo=dinfo+ ";"+ fun::LongStr(MaxMemoryCpu)+ ";"+ fun::LongStr(MaxMemoryGpu);
  const unsigned nout=GetOutPosCount()+GetOutRhopCount()+GetOutMoveCount();
  dinfo=dinfo+ ";"+ fun::IntStr(Nstep)+ ";"+ fun::IntStr(Part)+ ";"+ fun::UintStr(nout);
  dinfo=dinfo+ ";"+ fun::UintStr(MaxParticles)+ ";"+ fun::UintStr(MaxCells);
  dinfo=dinfo+ ";"+ Hardware+ ";"+ GetStepName(TStep)+ ";"+ GetKernelName(TKernel)+ ";"+ GetViscoName(TVisco)+ ";"+ fun::FloatStr(Visco);
  dinfo=dinfo+ ";"+ fun::FloatStr(DeltaSph)+ ";"+ fun::UintStr(ShepardSteps)+ ";"+ fun::FloatStr(TimeMax);
  dinfo=dinfo+ ";"+ fun::UintStr(CaseNbound)+ ";"+ fun::UintStr(CaseNfixed)+ ";"+ fun::FloatStr(H);
  char rhopcad[256];
  if(RhopOut)sprintf(rhopcad,"(%G-%G)",RhopOutMin,RhopOutMax); else sprintf(rhopcad,"None");
  dinfo=dinfo+ ";"+ rhopcad+ ";"+ fun::UintStr(GetOutRhopCount())+ ";"+ fun::UintStr(GetOutMoveCount())+ ";"+ GetNameCellMode(CellMode)+ detplus;
}

//==============================================================================
/// Generates file Run.csv with summary of execution.
//==============================================================================
void JSph::SaveRes(float tsim,float ttot,const std::string &headplus,const std::string &detplus){
  const char* met="SaveRes";
  string fname=DirOut+"Run.csv";
  ofstream pf;
  pf.open(fname.c_str());
  if(pf){
    string hinfo,dinfo;
    GetResInfo(tsim,ttot,headplus,detplus,hinfo,dinfo);
    pf << hinfo << endl << dinfo << endl;
    if(pf.fail())RunException(met,"Failed writing to file.",fname);
    pf.close();
  }
  else RunException(met,"File could not be opened.",fname);
}

//==============================================================================
/// Shows summary of execution.
//==============================================================================
void JSph::ShowResume(bool stop,float tsim,float ttot,bool all,std::string infoplus){
  char cad[512];
  sprintf(cad,"\n[Simulation %s  %s]",(stop? "INTERRUPTED": "finished"),fun::GetDateTime().c_str());  Log->Print(cad);
  if(all){
    sprintf(cad,"DTs adjusted to DtMin.............: %d",DtModif);  Log->Print(cad);
    const unsigned nout=GetOutPosCount()+GetOutRhopCount()+GetOutMoveCount();
    sprintf(cad,"Excluded particles................: %d",nout);  Log->Print(cad);
    if(GetOutRhopCount()){ sprintf(cad,"Excluded particles due to RhopOut.: %u",GetOutRhopCount());  Log->Print(cad); }
    if(GetOutMoveCount()){    sprintf(cad,"Excluded particles due to Velocity: %u",GetOutMoveCount());  Log->Print(cad); }
    if(GetOutInitialCount()){ sprintf(cad,"Excluded particles initially......: %u",GetOutInitialCount()); Log->Print(cad); }
  }
  sprintf(cad,"Total Runtime.....................: %f sec.",ttot);    Log->Print(cad);
  sprintf(cad,"Simulation Runtime................: %f sec.",tsim);    Log->Print(cad);
  if(all){
    float tseg=tsim/TimeStep;
    float nstepseg=float(Nstep)/tsim;
    sprintf(cad,"Time per second of simulation.....: %f sec.",tseg);    Log->Print(cad);
    sprintf(cad,"Steps per second..................: %f",nstepseg);     Log->Print(cad);
    sprintf(cad,"Steps of simulation...............: %d",Nstep);        Log->Print(cad);
    sprintf(cad,"PART files........................: %d",Part-PartIni); Log->Print(cad);
    while(!infoplus.empty()){
      string lin=fun::StrSplit("#",infoplus);
      if(!lin.empty()){
        string tex=fun::StrSplit("=",lin);
        string val=fun::StrSplit("=",lin);
        while(tex.size()<33)tex=tex+".";
        Log->Print(tex+": "+val);
      }
    }
  }
  sprintf(cad,"Maximum number of particles.......: %u",MaxParticles); Log->Print(cad);
  sprintf(cad,"Maximum number of cells...........: %u",MaxCells);     Log->Print(cad);
  //SL: Changed %u to %lld as "MaxMemoryXXX" variables are long long int so %u throws a warning
  sprintf(cad,"CPU Memory........................: %lld (%.2f MB)",MaxMemoryCpu,double(MaxMemoryCpu)/(1024*1024));  Log->Print(cad);
  if(MaxMemoryGpu){ sprintf(cad,"GPU Memory........................: %lld (%.2f MB)",MaxMemoryGpu,double(MaxMemoryGpu)/(1024*1024));  Log->Print(cad); }
}

//==============================================================================
/// Returns the name of the algorithm in text format.
//==============================================================================
std::string JSph::GetStepName(TpStep tstep){
  string tx;
  if(tstep==STEP_Verlet)tx="Verlet";
  else if(tstep==STEP_Symplectic)tx="Symplectic";
  else tx="???";
  return(tx);
}

//==============================================================================
/// Returns the name of the kernel function in text format.
//==============================================================================
std::string JSph::GetKernelName(TpKernel tkernel){
  string tx;
  if(tkernel==KERNEL_Cubic)tx="Cubic";
  else if(tkernel==KERNEL_Wendland)tx="Wendland";
  else tx="???";
  return(tx);
}

//==============================================================================
/// Returns the name of the viscosity treatment in text format.
//==============================================================================
std::string JSph::GetViscoName(TpVisco tvisco){
  string tx;
  if(tvisco==VISCO_Artificial)tx="Artificial";
  else if(tvisco==VISCO_LaminarSPS)tx="LaminarSPS"; 
  else if(tvisco==VISCO_SumSPS)tx="SumSPS"; 
  else tx="???";
  return(tx);
}

//==============================================================================
/// Returns the name of deltaSPH approach in text format.
//==============================================================================
std::string JSph::GetDeltaSphName(TpDeltaSph tdelta){
  string tx;
  if(tdelta==DELTA_None)tx="None";
  //else if(tdelta==DELTA_Basic)tx="Basic";
  else if(tdelta==DELTA_DBC)tx="DBC";
  else if(tdelta==DELTA_DBCExt)tx="DBCExt";
  else tx="???";
  return(tx);
}

//==============================================================================
/// Returns string with the name of the timer and its value.
//==============================================================================
std::string JSph::TimerToText(const std::string &name,float value){
  string ret=name;
  while(ret.length()<33)ret+=".";
  return(ret+": "+fun::FloatStr(value/1000)+" sec.");
}

//==============================================================================
/// Stores CSV file with particle data.
//==============================================================================
/*void JSph::DgSaveCsvParticlesCpu(std::string filename,int numfile,unsigned pini,unsigned pfin,std::string head,const tfloat3 *pos,const unsigned *idp,const tfloat3 *vel,const float *rhop,const float *ar,const tfloat3 *ace,const tfloat3 *vcorr){
  const char met[]="DgSaveCsvParticlesCpu";
  int mpirank=Log->GetMpiRank();
  if(mpirank>=0)filename=string("p")+fun::IntStr(mpirank)+"_"+filename;
  if(numfile>=0){
    string ext=fun::GetExtension(filename);
    filename=fun::GetWithoutExtension(filename);
    char cad[64];
    sprintf(cad,"%04d",numfile);
    filename=filename+cad+"."+ext;
  }
  filename=DirOut+filename;
  //-Generates CSV file.
  ofstream pf;
  pf.open(filename.c_str());
  if(pf){
    if(!head.empty())pf << head << endl;
    pf << "Num";
    if(idp)pf << ";Idp";
    if(pos)pf << ";PosX;PosY;PosZ";
    if(vel)pf << ";VelX;VelY;VelZ";
    if(rhop)pf << ";Rhop";
    if(ar)pf << ";Ar";
    if(ace)pf << ";AceX;AceY;AceZ";
    if(vcorr)pf << ";VcorrX;VcorrY;VcorrZ";
    pf << endl;
    const char fmt1[]="%f"; //="%24.16f";
    const char fmt3[]="%f;%f;%f"; //="%24.16f;%24.16f;%24.16f";
    for(unsigned p=pini;p<pfin;p++){
      pf << fun::UintStr(p-pini);
      if(idp)pf << ";" << fun::UintStr(idp[p]);
      if(pos)pf << ";" << fun::Float3Str(pos[p],fmt3);
      if(vel)pf << ";" << fun::Float3Str(vel[p],fmt3);
      if(rhop)pf << ";" << fun::FloatStr(rhop[p],fmt1);
      if(ar)pf << ";" << fun::FloatStr(ar[p],fmt1);
      if(ace)pf << ";" << fun::Float3Str(ace[p],fmt3);
      if(vcorr)pf << ";" << fun::Float3Str(vcorr[p],fmt3);
      pf << endl;
    }
    if(pf.fail())RunException(met,"Failed writing to file.",filename);
    pf.close();
  }
  else RunException(met,"File could not be opened.",filename);
}*/


