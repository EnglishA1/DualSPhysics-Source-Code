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

/// \file JSphGpuSingle.cpp \brief Implements the class \ref JSphGpuSingle.

#include "JSphGpuSingle.h"
#include "JSphGpu_ker.h"

//#include "GMultiGpu_ker.h"

#include "Functions.h"
#include "JSphMotion.h"
#include "JCellDivGpuSingle.h"
#include "JPtxasInfo.h"
#include "JGpuArrays.h"
#include "JPartsOut.h"
#include "JPartsLoad.h"
#include "JPeriodicGpu.h"

#include "JPartsLoad4.h"

using namespace std;
//==============================================================================
/// Constructor.
//==============================================================================
JSphGpuSingle::JSphGpuSingle(){
  ClassName="JSphGpuSingle";
  CellDivSingle=NULL;
  PartsLoaded=NULL;
  PartsLoaded4=NULL;
  PeriZone=NULL;
}

//==============================================================================
/// Destructor.
//==============================================================================
JSphGpuSingle::~JSphGpuSingle(){
  delete CellDivSingle; CellDivSingle=NULL;
  delete PartsLoaded;   PartsLoaded=NULL;
  delete PartsLoaded4;   PartsLoaded4=NULL;
  delete PeriZone;      PeriZone=NULL;
}

//==============================================================================
/// Returns the allocated memory in CPU.
//==============================================================================
long long JSphGpuSingle::GetAllocMemoryCpu()const{  
  long long s=JSphGpu::GetAllocMemoryCpu();
  //-Allocated in other objects.
  if(CellDivSingle)s+=CellDivSingle->GetAllocMemoryCpu();
  if(PartsLoaded)s+=PartsLoaded->GetAllocMemory();

  //if(PartsLoaded4)s+=PartsLoaded4->GetAllocMemory();

  return(s);
}

//==============================================================================
/// Returns the allocated memory in GPU.
//==============================================================================
long long JSphGpuSingle::GetAllocMemoryGpu()const{  
  long long s=JSphGpu::GetAllocMemoryGpu();
  //-Allocated in other objects.
  if(CellDivSingle)s+=CellDivSingle->GetAllocMemoryGpu();
  if(PeriZone)s+=PeriZone->GetAllocMemoryGpu();
  return(s);
}

//==============================================================================
/// Returns the GPU memory allocated or used for particles.
//==============================================================================
long long JSphGpuSingle::GetMemoryGpuNp()const{
  long long s=JSphGpu::GetAllocMemoryGpu();
  //-Allocated in other objects.
  if(CellDivSingle)s+=CellDivSingle->GetAllocMemoryGpuNp();
  if(PeriZone)s+=PeriZone->GetAllocMemoryGpuNp();
  return(s);
}

//==============================================================================
/// Returns the GPU memory allocated or used for cells.
//==============================================================================
long long JSphGpuSingle::GetMemoryGpuNct()const{
  long long s=CellDivSingle->GetAllocMemoryGpuNct();
  if(PeriZone)s+=PeriZone->GetAllocMemoryGpuNct();
  return(CellDivSingle->GetAllocMemoryGpuNct());
}

//==============================================================================
/// Updates the maximum values of memory, particles and cells.
//==============================================================================
void JSphGpuSingle::UpdateMaxValues(){
  MaxParticles=max(MaxParticles,Np);
  if(CellDivSingle)MaxCells=max(MaxCells,CellDivSingle->GetNct());
  long long m=GetAllocMemoryCpu();
  MaxMemoryCpu=max(MaxMemoryCpu,m);
  m=GetAllocMemoryGpu();
  MaxMemoryGpu=max(MaxMemoryGpu,m);
}

//==============================================================================
/// Loads the configuration of the execution.
//==============================================================================
void JSphGpuSingle::LoadConfig(JCfgRun *cfg){
  PtxasFile=cfg->PtxasFile;
  JSph::LoadConfig(cfg);     //-Loads general configuration.
}

//==============================================================================
/// Loads particles of the case to be processesed.
//==============================================================================
void JSphGpuSingle::LoadCaseParticles(){
  Log->Print("Loading initial state of particles...");
  
  
  PartsLoaded=new JPartsLoad;
  PartsLoaded4=new JPartsLoad4;
  
  PartsLoaded4->LoadParticles4(DirCase,CaseName,PartBegin,PartBeginDir);
  PartsLoaded4->CheckConfig4(CaseNp,CaseNfixed,CaseNmoving,CaseNfloat,CaseNfluid,PeriX,PeriY,PeriZ);
  sprintf(Cad,"Loaded particles: %u ",PartsLoaded4->GetCount()); Log->Print(Cad);
  SetOutInitialCount(PartsLoaded4->GetCount()-PartsLoaded->GetCountOk());
  //-Recovers information of the loaded particles.
  Simulate2D=PartsLoaded4->GetSimulate2D();
  if(Simulate2D&&PeriY)RunException("LoadCaseParticles","Periodic conditions in Y with 2D simulations can not be used.");
  
  /*
  //v3
  const float bordermax=H*BORDER_MAP;
  const float peribordermax=Dp/2.f;
  const tfloat3 border=TFloat3((PeriX? peribordermax: bordermax),(PeriY? peribordermax: bordermax),(PeriZ? peribordermax: bordermax));
  PartsLoaded->GetLimits(border,border,MapPosMin,MapPosMax);
  if(!PartBegin||PartsLoaded->LoadPartBi2())ResizeMapLimits();
  if(PartBegin)PartBeginTimeStep=PartsLoaded->GetPartBeginTimeStep();
  sprintf(Cad,"MapPos=(%f,%f,%f)-(%f,%f,%f)",MapPosMin.x,MapPosMin.y,MapPosMin.z,MapPosMax.x,MapPosMax.y,MapPosMax.z); Log->Print(Cad);
  Log->Print("**Initial state of particles is loaded");
  */

  //4
  CasePosMin=PartsLoaded4->GetCasePosMin();
  CasePosMax=PartsLoaded4->GetCasePosMax();

  //-Calcula limites reales de la simulacion.
  //-Computes the current limits of the simulation.
  if(PartsLoaded4->MapSizeLoaded())PartsLoaded4->GetMapSize(MapRealPosMin,MapRealPosMax);
  else{
    PartsLoaded4->CalculeLimits(double(H)*BORDER_MAP,Dp/2.,PeriX,PeriY,PeriZ,MapRealPosMin,MapRealPosMax);
    ResizeMapLimits();
  }
  if(PartBegin){
    double(PartBeginTimeStep)=PartsLoaded4->GetPartBeginTimeStep();
    PartBeginTotalNp=PartsLoaded4->GetPartBeginTotalNp();
  }
  Log->Print(string("MapRealPos(final)=")+fun::Double3gRangeStr(MapRealPosMin,MapRealPosMax));

  
  MapRealSize=MapRealPosMax-MapRealPosMin;
  MapPosMin=ToTFloat3(MapRealPosMin);
  MapPosMax=ToTFloat3(MapRealPosMax);

  Log->Print("**Initial state of particles is loaded");
  

  //-Configures limits of periodic axis.
  if(PeriX)PeriXinc.x=MapPosMin.x-MapPosMax.x;
  if(PeriY)PeriYinc.y=MapPosMin.y-MapPosMax.y;
  if(PeriZ)PeriZinc.z=MapPosMin.z-MapPosMax.z;
}

//==============================================================================
/// Configuration of the current domain.
//==============================================================================
void JSphGpuSingle::ConfigDomain(){
  const char* met="ConfigDomain";
  //-Allocates memory in CPU and locates particle data in the variables of the domain.
  AllocCpuMemoryParticles(CaseNp);
  if(CaseNp!=PartsLoaded4->GetCount())RunException(met,"The number of particles is invalid.");

  tdouble3 *auxpos=new tdouble3[CaseNp];
  tfloat4  *Velrhop=new tfloat4[CaseNp];
  memcpy(Idp,PartsLoaded4->GetIdp(),CaseNp*sizeof(unsigned));
  memcpy(auxpos,PartsLoaded4->GetPos(),CaseNp*sizeof(tdouble3));
  //memcpy(Pos,PartsLoaded->GetPos(),CaseNp*sizeof(tfloat3));
  memcpy(Velrhop,PartsLoaded4->GetVelRhop(),sizeof(tfloat4)*CaseNp);

  //I had to do this, not very nice but it works, sory Jose!
  for(unsigned p=0;p<CaseNp;p++){
	  Pos[p]=ToTFloat3(auxpos[p]);
      tfloat4 vr=Velrhop[p];
      Vel[p]=TFloat3(vr.x,vr.y,vr.z);
      Rhop[p]=vr.w;
  }
  delete[] auxpos;
  delete[] Velrhop;
  
  //memcpy(Vel,PartsLoaded->GetVel(),CaseNp*sizeof(tfloat3));
  //memcpy(Rhop,PartsLoaded->GetRhop(),CaseNp*sizeof(float));

  /*//debug
  std::ofstream outpart;
  outpart.open ("test.txt");
  for(unsigned p=0;p<CaseNp;p++){
	//printf("Pos.x=%f Pos.z=%f \n",Pos[p].x,Pos[p].z);
	outpart << auxpos[p].x <<"\t"<< auxpos[p].z << endl;
  }
  outpart.close(); */

  memset(Press,0,CaseNp*sizeof(float));
  memset(Vtau,0,CaseNp*sizeof(float));
  memset(Idpm,0,CaseNp*sizeof(unsigned));
  
  //Log->Print("ConfigDomain");
  //getchar();

  //-Loads code of particles.
  //LoadCodeParticles(CaseNp,PartsLoaded4->GetCountOut(),Idp,Code);
  LoadCodeParticles(CaseNp,Idp,Code);
  //-Releases memory of PartsLoaded.
  delete PartsLoaded; PartsLoaded=NULL;
  //-Computes number of particles.
  Np=CaseNp; Npb=CaseNpb; NpbOk=Npb;
  
  //*******************************adjusts rhop and idm for multi*****************************
  //-Looking fluid limits
  float minx=Pos[Npb].x,miny=Pos[Npb].y,maxx=minx,maxy=miny;
  //printf("\n minx=%f ,miny=%f ,maxx=%f ,maxy=%f \n",minx ,miny ,maxx ,maxy);
  for(unsigned p=Npb;p<Np;p++){
    const tfloat3 *ps=(Pos+p);
	if(minx>ps->x)minx=ps->x;
	if(miny>ps->y)miny=ps->y;
	if(maxx<ps->x)maxx=ps->x;
	if(maxy<ps->y)maxy=ps->y;
  }

  printf("Domainfluid:(%f,%f)-(%f,%f)  dp:%f\n",minx,miny,maxx,maxy,Dp);
  const float ovdp=1/Dp;
  const int ncx=int((maxx-minx)*ovdp)+2,ncy=int((maxy-miny)*ovdp)+2;
  const int nc=ncx*ncy;
  printf("Domainfluid> ncx:%d ncy:%d  nc:%d\n",ncx,ncy,nc);
  float epsilon=Dp/10;
  float mindomx=minx-epsilon,mindomy=miny-epsilon;
  float sizedomx=(maxx-minx)+2*epsilon,sizedomy=(maxy-miny)+2*epsilon;
  //-Calculate x-column of each particle and counted.
  int* partrow=new int[Np];
  int* inrow=new int[nc];
  memset(inrow,0,sizeof(int)*nc);
  int npin=0;
  for(unsigned p=0;p<Np;p++){
    float px=Pos[p].x,py=Pos[p].y;
	float dx=px-minx+epsilon,dy=py-miny+epsilon;
	if(px>mindomx&&py>mindomy&&dx<sizedomx&&dy<sizedomy){//In fluid domain.
      unsigned cx=unsigned(dx*ovdp),cy=unsigned(dy*ovdp);
	  int row=cy*ncx+cx;
	  partrow[p]=row;
	  inrow[row]++;
	  npin++;
	}
	else partrow[p]=-1;
  }

  printf("npin:%d\n",npin);
  //-Sets initial position of columns.
  int* beginrow=new int[nc+1];
  beginrow[0]=0;
  for(int row=0;row<nc;row++){
	beginrow[row+1]=beginrow[row]+inrow[row];
	inrow[row]=0;
  }

  //-Place the position (Pos []) of particles in the column where it belongs.
  unsigned* partpos=new unsigned[npin];
  for(unsigned p=0;p<Np;p++){
    int row=partrow[p];
	if(row>=0){
	  partpos[beginrow[row]+inrow[row]]=p;
	  inrow[row]++;
	}
  }

  //-Adjusts the density of fluid particles according to the above need.
  int* nphasedeep=new int[Phases];
  for(int row=0;row<nc;row++){
    int ini=beginrow[row];
    int fin=beginrow[row+1];
	for(int c=ini;c<fin;c++)if(partpos[c]>=Npb){//-Calculate the particles of each type above it (only for fluid).
      unsigned ppos=partpos[c];
      float pz=Pos[ppos].z;
	  //-Estimated maximum depth to the nearest boundary.
      float maxz=FLT_MAX;
	  for(int c2=ini;c2<fin;c2++)if(partpos[c2]<Npb){
        unsigned ppos2=partpos[c2];
        float pz2=Pos[ppos2].z;
		if(pz<pz2&&pz2<maxz)maxz=pz2;
	  }
	  //-Calculate fluid particles over each type up to nearest boundary.
	  memset(nphasedeep,0,sizeof(int)*Phases);
	  for(int c2=ini;c2<fin;c2++)if(partpos[c2]>=Npb){
        unsigned ppos2=partpos[c2];
        float pz2=Pos[ppos2].z;
		if(pz<pz2&&pz2<maxz){
		  unsigned id2=Idp[ppos2];
		  int cphase;
		  for(cphase=0;cphase<int(Phases)&&(id2<PhaseCte[cphase].idbegin||id2>=PhaseCte[cphase].idbegin+PhaseCte[cphase].count);cphase++);	
		  if(cphase>=int(Phases))RunException(met,"Fluid particle unrecognized.");
		  nphasedeep[cphase]++;
		}
	  }
	  //-Adjust according to the particle density to support.
	  float sumrhop0=0;
	  for(int cphase=0;cphase<int(Phases);cphase++)sumrhop0+=PhaseArray[cphase].rho_ph*nphasedeep[cphase]*Dp;
	  unsigned id=Idp[ppos];
	  int cphase;
	  for(cphase=0;cphase<int(Phases)&&(id<PhaseCte[cphase].idbegin||id>=PhaseCte[cphase].idbegin+PhaseCte[cphase].count);cphase++);	
	  Rhop[ppos]=PhaseArray[cphase].rho_ph* pow((1+sumrhop0/PhaseArray[cphase].b_ph),1/PhaseArray[cphase].Gamma_ph);
	  Press[ppos]=PhaseArray[cphase].b_ph*(pow(Rhop[ppos]/PhaseArray[cphase].rho_ph,7)-1.f);
	  Viscop[ppos]=PhaseCte[cphase].visco;
	  Idpm[ppos]=cphase; //fluid idpm 0/1

	}
  }

  //Boundary idpm set to min desnity for now which is the fluid usualy
  float rhomin=FLT_MAX;
  unsigned c;
  for(unsigned cphase=0;cphase<int(Phases);cphase++) {
	  rhomin=min(PhaseArray[cphase].rho_ph,rhomin);
	  if (PhaseArray[cphase].rho_ph==rhomin)c=cphase;
  }
  //Boundaries are set according to the original DualSPH constants visco,rhop0 etc.
  for(unsigned p=0;p<Npb;p++){
	  Idpm[p]=c;
	  Viscop[p]=Visco;
	  Rhop[p]=Rhop0;
	  Press[p]=0.0f;
  }

  //-Frees dynamic memory the calculation of depth
  delete[] inrow;
  delete[] beginrow;
  delete[] partrow;
  delete[] partpos;
  delete[] nphasedeep;
  //**********************************************************************
  
  //-Applies configuration of CellOrder.
  ConfigCellOrder(CellOrder,Np,Pos,Vel);

  //-Allocates memory in GPU and copies data to GPU.
  AllocGpuMemoryParticles(Np,0);
  ReserveBasicGpuArrays();
  ParticlesDataUp(Np);

  //-Creates object for CellDiv in GPU and selects a valid cellmode.
  CellDivSingle=new JCellDivGpuSingle(Stable,Log,DirOut,PeriActive,TVisco==VISCO_LaminarSPS,!PeriActive,MapPosMin,MapPosMax,Dosh,CaseNbound,CaseNfixed,CaseNpb,CellOrder);
  //-Selects a valid cellmode.
  if(CellMode==CELLMODE_None)CellMode=(Simulate2D? CELLMODE_2H: CELLMODE_None);
  CellDivSingle->ConfigInit(CellMode,CaseNp,CaseNpb,RhopOut,RhopOutMin,RhopOutMax);
  CellMode=CellDivSingle->GetCellMode();
  Hdiv=CellDivSingle->GetHdiv();
  Scell=CellDivSingle->GetScell();
  MapCells=CellDivSingle->GetMapCells();
  MovLimit=Scell*0.9f;
  Log->Print(fun::VarStr("CellMode",string(GetNameCellMode(CellMode))));
  Log->Print(fun::VarStr("Hdiv",Hdiv));
  Log->Print(string("MapCells=(")+fun::Uint3Str(OrderDecode(MapCells))+")");
  SaveMapCellsVtk(Scell);
  ConfigCellDiv((JCellDivGpu*)CellDivSingle);
  ConfigBlockSizes(false,PeriActive!=0);
  //-Initialises the use of periodic conditions.
  if(PeriActive)PeriInit();
  //-Reorders particles according to cells.

  ConfigSaveData(0,1,"");
  BoundChanged=true;
  RunCellDivide(false);
}

//==============================================================================
/// Initialises the use of periodic conditions.
//==============================================================================
void JSphGpuSingle::PeriInit(){
  PeriZone=new JPeriodicGpu(Stable,Log,DirOut,MapPosMin,MapPosMax,(JCellDivGpu*)CellDivSingle);
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
void JSphGpuSingle::PeriCheckPosition(){
  TmgStart(Timers,TMG_NlPeriCheck);
  PeriZone->CheckPositionAll(BoundChanged,Posg);
  TmgStop(Timers,TMG_NlPeriCheck);
}

//==============================================================================
/// Configures execution mode in GPU.
//==============================================================================
void JSphGpuSingle::ConfigRunMode(const std::string &preinfo){
  #ifndef WIN32
    const int len=128; char hname[len];
    gethostname(hname,len);
    JSphGpu::ConfigRunMode(preinfo+", HostName:"+hname);
  #else
    JSphGpu::ConfigRunMode(preinfo);
  #endif
}

//==============================================================================
/// Executes CellDiv of particles in cells.
//==============================================================================
void JSphGpuSingle::RunCellDivide(bool symvars){
  //-Checks positions of the particles in the periodic edges.
  if(PeriActive)PeriCheckPosition();
  //-Initiates Divide.
  CellDivSingle->Divide(BoundChanged,Posg,Rhopg,Codeg,Timers);
  //-Orders particle data.
  TmgStart(Timers,TMG_NlSortData);
  {
    unsigned* idpg=GpuArrays->ReserveUint();
    word* codeg=GpuArrays->ReserveWord();
    float3* posg=GpuArrays->ReserveFloat3();
    float3* velg=GpuArrays->ReserveFloat3();
    float* rhopg=GpuArrays->ReserveFloat();
	//multi
	float* pressg=GpuArrays->ReserveFloat();
	float* viscopg=GpuArrays->ReserveFloat();
	unsigned* idpmg=GpuArrays->ReserveUint();
	//reorder
	CellDivSingle->SortBasicArrays(Idpg,Posg,Velg,Rhopg,Codeg,Pressg,Viscopg,Idpmg,  idpg,posg,velg,rhopg,codeg,pressg,viscopg,idpmg);
	//swap bak to arrays
    swap(Idpg,idpg);     GpuArrays->Free(idpg);
    swap(Codeg,codeg);   GpuArrays->Free(codeg);
    swap(Posg,posg);     GpuArrays->Free(posg);
    swap(Velg,velg);     GpuArrays->Free(velg);
    swap(Rhopg,rhopg);   GpuArrays->Free(rhopg);
	//multi
	swap(Pressg,pressg);     GpuArrays->Free(pressg);
	swap(Viscopg,viscopg);   GpuArrays->Free(viscopg);
	swap(Idpmg,idpmg);       GpuArrays->Free(idpmg);

  }    
  if(TStep==STEP_Verlet){
    float* rhopg=GpuArrays->ReserveFloat();
    float3* velg=GpuArrays->ReserveFloat3();
    CellDivSingle->SortDataArrays(RhopM1g,VelM1g,rhopg,velg);
    swap(RhopM1g,rhopg);    GpuArrays->Free(rhopg);
    swap(VelM1g,velg);      GpuArrays->Free(velg);
  }
  else if(TStep==STEP_Symplectic&&symvars){
    float* rhopg=GpuArrays->ReserveFloat();
    float3* posg=GpuArrays->ReserveFloat3();
    float3* velg=GpuArrays->ReserveFloat3();
    CellDivSingle->SortDataArrays(RhopPreg,PosPreg,VelPreg,rhopg,posg,velg);
    swap(RhopPreg,rhopg);   GpuArrays->Free(rhopg);
    swap(PosPreg,posg);     GpuArrays->Free(posg);
    swap(VelPreg,velg);     GpuArrays->Free(velg);
  }
  if(TVisco==VISCO_LaminarSPS){  
    tsymatrix3f* taug=GpuArrays->ReserveSymatrix3f();
    CellDivSingle->SortDataArrays(Taug,taug);
    swap(Taug,taug);       GpuArrays->Free(taug);
  }

  if(TVisco==VISCO_SumSPS){  
    tsymatrix3f* taug=GpuArrays->ReserveSymatrix3f();
    CellDivSingle->SortDataArrays(Taug,taug);
    swap(Taug,taug);       GpuArrays->Free(taug);

	tsymatrix3f* vtaug=GpuArrays->ReserveSymatrix3f();
	CellDivSingle->SortDataArrays(Vtaug,vtaug);
	swap(Vtaug,vtaug);       GpuArrays->Free(vtaug);

  }


  //-Recovers data of CellDiv and updates number of particles of each type.
  Np=CellDivSingle->GetNp();
  NpbOk=Npb-CellDivSingle->GetNpbIgnore();
  TmgStop(Timers,TMG_NlSortData);
  //-Manages excluded particles (bound and fluid).
  TmgStart(Timers,TMG_NlOutCheck);
  unsigned npout=CellDivSingle->GetNpOut();
  if(npout){
    ParticlesDataDown(npout,Np,true,true);
    CellDivSingle->CheckParticlesOut(npout,Idp,Pos,Rhop,Code);
    PartsOut->AddParticles(npout,Idp,Pos,Vel,Rhop,CellDivSingle->GetNpfOutRhop(),CellDivSingle->GetNpfOutMove());
  }
  TmgStop(Timers,TMG_NlOutCheck);
  BoundChanged=false;
}

//==============================================================================
/// Generates periodic zones and computes interaction.
//==============================================================================
void JSphGpuSingle::PeriInteraction(TpInter tinter){
  const bool xsph=(tinter==INTER_Forces);
  const unsigned bsfluid=(xsph? BlockSizes.forcesfluid_peri: BlockSizes.forcesfluidcorr_peri);
  const unsigned bsbound=BlockSizes.forcesbound_peri;
  const unsigned nzone=PeriZone->GetZonesCount();
  for(unsigned c=0;c<nzone;c++){
    TmgStart(Timers,TMG_NlPeriPrepare); 
    PeriZone->PrepareForces(c,PosPresg,Codeg,VelRhopg,Idpg,Taug);
    TmgStop(Timers,TMG_NlPeriPrepare);
    TmgStart(Timers,TMG_CfPeriForces);
    if(CaseNfloat){     //-Checks if there are floating particles in the periodic zone.
      unsigned idfloat=cusph::FtFindFirstFloating(PeriZone->GetListNp(),PeriZone->GetListNpb(),PeriZone->GetList(),PeriZone->GetListFluidIni(),Codeg,Idpg);
      if(idfloat!=UINT_MAX){
        sprintf(Cad,"Particle id:%u (from a floating body) in Periodicity region.",idfloat);
        RunException("PeriInteraction",Cad);
      }
    }
    cusph::InteractionPeri_Forces(TDeltaSph,TKernel,TVisco,xsph,CellMode,bsbound,bsfluid,PeriZone->GetListNp(),PeriZone->GetListNpb(),PeriZone->GetList(),PeriZone->GetListBoundIni(),PeriZone->GetListFluidIni(),PeriZone->GetListCellPart(),PeriZone->GetNc1(),PeriZone->GetBoxFluid(),PeriZone->GetBeginEndCell(),PeriZone->GetPosPres(),PeriZone->GetVelRhop(),PeriZone->GetIdp(),PeriZone->GetTau(),PosPresg,VelRhopg,Idpg,Taug,ViscDtg,Arg,Aceg,VelXcorg,Csphg,Deltag,Simulate2D);
    TmgStop(Timers,TMG_CfPeriForces);
  }
}

//==============================================================================
/// Interaction for force computation.
//==============================================================================
void JSphGpuSingle::Interaction_Forces(TpInter tinter){
  const char met[]="Interaction_Forces";
  PreInteraction_Forces(tinter);
  TmgStart(Timers,TMG_CfForces);

  const bool xsph=(tinter==INTER_Forces);
  const unsigned bsfluid=(xsph? BlockSizes.forcesfluid: BlockSizes.forcesfluidcorr);
  const unsigned bsbound=BlockSizes.forcesbound;
  
  //CudaArrayPrintFloat(Np,Viscopg,Idpg);
  //CudaArrayPrintFloat3(Np,Aceg);
  //CudaArrayPrintFloat4(Np,);
  //CudaArrayPrintTsym3f(Np,Vtaug);

  if(WithFloating)cusph::FtInteraction_Forces(TDeltaSph,TKernel,TVisco,xsph,CellMode,bsbound,bsfluid,Np,Npb,NpbOk,CellDivSingle->GetNcells(),CellDivSingle->GetCellPart(),CellDivSingle->GetBeginCell(),PosPresg,VelRhopg,Idpg,Taug,Codeg,FtoMasspg,ViscDtg,Arg,Aceg,VelXcorg,Csphg,Deltag,Simulate2D);
    
  else if (Multiphase)cusph::MultiInteraction_Forces  (TDeltaSph,TKernel,TVisco,xsph,CellMode,bsbound,bsfluid,Np,Npb,NpbOk,CellDivSingle->GetNcells(),CellDivSingle->GetCellPart(),CellDivSingle->GetBeginCell(),PosPresg,VelRhopg,Idpg,Taug,Vtaug,ViscDtg,Arg,Aceg,VelXcorg,Csphg,Deltag,Idpmg,Viscopg,CVg,Simulate2D);
  
  else            cusph::Interaction_Forces  (TDeltaSph,TKernel,TVisco,xsph,CellMode,bsbound,bsfluid,Np,Npb,NpbOk,CellDivSingle->GetNcells(),CellDivSingle->GetCellPart(),CellDivSingle->GetBeginCell(),PosPresg,VelRhopg,Idpg,Taug,ViscDtg,Arg,Aceg,VelXcorg,Csphg,Deltag,Simulate2D);
  
  if(PeriActive){
    TmgStop(Timers,TMG_CfForces);
    PeriInteraction(tinter);
    TmgStart(Timers,TMG_CfForces);
  }
  if(Deltag)cusph::AddDelta(Np-Npb,Deltag+Npb,Arg+Npb);    //-Applies Delta-SPH to Arg[].
  CheckCudaError(met,"Failed while executing kernels of interaction.");
  
  //only for single phase
  if(TVisco==VISCO_LaminarSPS)cusph::SPSCalcTau(WithFloating,Np,Npb,SpsSmag,SpsBlin,Rhopg,Codeg,Csphg,Taug);  //(kinematic here dynamic in SUmVsic)

  //Remove Ace for unyielded material
  //if (Multiphase)cusph::YieldResetAce(Np,Npb,Vtaug,Viscopg,Aceg,Idpmg,Velg,Arg);
  
  //Debug
  //CudaArrayPrintFloat(Np,Rhopg);

  //Reduce for Cs, Forces and Visocity
  if(Np)ViscDtMax=cusph::ReduMaxFloat(Np,0,ViscDtg,CellDivSingle->GetAuxMem(cusph::ReduMaxFloatSize(Np)));
  //forces
  cusph::CalcFa2(Np,Aceg,ViscDtg);
  ForceDtMax=cusph::ReduMaxFloat(Np,0,ViscDtg,CellDivSingle->GetAuxMem(cusph::ReduMaxFloatSize(Np)));
  //viscous
  for (unsigned c=0;c<Phases;c++)ViscoDtMax=max(PhaseCte[c].visco,ViscoDtMax);
  ViscoDtMax=0.5f/ViscoDtMax*cusph::ReduMaxFloat(Np,0,Rhopg,CellDivSingle->GetAuxMem(cusph::ReduMaxFloatSize(Np))); // this is wrong, need to reduce rho, htaHBP but with an upper limit on HBP

  TmgStop(Timers,TMG_CfForces);
  CheckCudaError(met,"Failed in reduction of viscdt.");

  //if(0){  //dbg
  //  char cad[512];
  //  float *arh=new float[Np];
  //  float3 *aceh=new float3[Np];
  //  float3 *vcorh=new float3[Np];
  //  unsigned *idph=new unsigned[Np];
  //  cudaMemcpy(arh,Arg,sizeof(float)*Np,cudaMemcpyDeviceToHost);
  //  cudaMemcpy(aceh,Aceg,sizeof(float3)*Np,cudaMemcpyDeviceToHost);
  //  cudaMemcpy(vcorh,VelXcorg,sizeof(float3)*Np,cudaMemcpyDeviceToHost);
  //  cudaMemcpy(idph,Idpg,sizeof(unsigned)*Np,cudaMemcpyDeviceToHost);
  //  for(unsigned p=0;p<Np;p++){
  //    unsigned id=idph[p];
  //    if(id==225||id==264){
  //      sprintf(cad,"particle[%u]> idp:%u  ar:%f  ace:(%f,%f,%f)",p,idph[p],arh[p],aceh[p].x,aceh[p].y,aceh[p].z); Log->Print(cad);
  //    }
  //    //sprintf(cad,"particle[%u]> idp:%u  ar:%f  vcor:(%f,%f,%f)",p,idph[p],arh[p],vcorh[p].x,vcorh[p].y,vcorh[p].z); Log->Print(cad);
  //  }
  //  Log->Print(" ");
  //  delete[] arh;
  //  delete[] aceh;
  //  delete[] vcorh;
  //  delete[] idph;
  //}
}

//==============================================================================
/// Computation of the step: Particle interaction and update of particle data
/// according to the forces computed in the interaction using VERLET.
//==============================================================================
float JSphGpuSingle::ComputeStep_Ver(bool rhopbound){
  Interaction_Forces(INTER_Forces);    //-Interaction.
  const float dt=DtVariable();         //-Computes new dt.
  ComputeVerlet(rhopbound,dt);         //-Updates particles using Verlet.
  if(CaseNfloat)RunFloating(dt,false); //-Processes floating bodies.
  PosInteraction_Forces();             //-Releases memory of interaction.
  return(dt);
}

//==============================================================================
/// Computation of the step: Particle interaction and update of particle data
/// according to the forces computed in the interaction using SYMPLECTIC.
//==============================================================================
float JSphGpuSingle::ComputeStep_Sym(bool rhopbound){
  const float dt=DtPre;
  //-Predictor.
  //-----------
  Interaction_Forces(INTER_Forces);         //-Interaction.
  const float ddt_p=DtVariable();           //-Computes dt of Predictor.
  ComputeSymplecticPre(rhopbound,dt);       //-Applies Symplectic-Predictor to particle data.
  if(CaseNfloat)RunFloating(dt*.5f,true);   //-Processes floating bodies.
  PosInteraction_Forces();                  //-Releases memory of interaction.
  //-Corrector.
  //-----------
  RunCellDivide(true);
  Interaction_Forces(INTER_ForcesCorr);     //-Interaction without VelXCor[].
  const float ddt_c=DtVariable();           //-Computes dt of Corrector.
  ComputeSymplecticCorr(rhopbound,dt);      //-Applies Symplectic-Corrector to particle data.
  if(CaseNfloat)RunFloating(dt,false);      //-Processes floating bodies.
  PosInteraction_Forces();                  //-Releases memory of interaction.

  DtPre=min(ddt_p,ddt_c);                   //-Computes dt for the following ComputeStep.
  return(dt);
}

//==============================================================================
/// Processes floating bodies.
//==============================================================================
void JSphGpuSingle::RunFloating(float dt2,bool predictor){
  if(TimeStep>FtPause){
    TmgStart(Timers,TMG_SuFloating);
    cudaMemset(FtRidpg,255,sizeof(unsigned)*CaseNfloat); //-Assigns UINT_MAX values.
    cusph::CalcRidp(Np-Npb,Npb,CaseNpb,CaseNpb+CaseNfloat,Idpg,FtRidpg);
    for(unsigned cf=0;cf<FtCount;cf++){
      StFloatingData *fobj=FtObjs+cf;
      //-Computes traslational and rotational velocities.
      tfloat3 face,fomegavel;
      {
        float3 *resultg=(float3 *)CellDivSingle->GetAuxMem(6);
        cusph::FtCalcOmega(fobj->count,fobj->begin-CaseNpb,Gravity,fobj->mass,FtRidpg,FtDistg,Aceg,resultg);
        tfloat3 result[2];
        cudaMemcpy(&result,resultg,sizeof(float3)*2,cudaMemcpyDeviceToHost);
        face=result[0];
        fomegavel=result[1];
      }
      //sprintf(Cad,"%u>> face:%s fomegavel:%s",Nstep,fun::Float3Str(face).c_str(),fun::Float3Str(fomegavel).c_str()); Log->PrintDbg(Cad);
      //-Recomputes values of floating.
      tfloat3 center=fobj->center;
      tfloat3 fvel=fobj->fvel;
      tfloat3 fomega=fobj->fomega;
      fomegavel.x/=fobj->inertia.x;
      fomegavel.y/=fobj->inertia.y;
      fomegavel.z/=fobj->inertia.z;    
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
      cusph::FtUpdate(predictor,fobj->count,fobj->begin-CaseNpb,dt2,center,fvel,fomega,FtRidpg,FtDistg,Posg,Velg);
      //-Stores data.
      if(!predictor){
        fobj->center=center;
        fobj->fvel=fvel;
        fobj->fomega=fomega;
      }
    }
    TmgStop(Timers,TMG_SuFloating);
  }
}

//==============================================================================
/// Generates periodic zones of interaction and computes Shepard interaction.
//==============================================================================
void JSphGpuSingle::PeriInteractionShepard(){
  const unsigned bsshepard=BlockSizes.shepard_peri;
  const unsigned nzone=PeriZone->GetZonesCount();
  for(unsigned c=0;c<nzone;c++){
    TmgStart(Timers,TMG_NlPeriPrepare); 
    PeriZone->PrepareShepard(c,PosVolg,Codeg);
    TmgStop(Timers,TMG_NlPeriPrepare);
    TmgStart(Timers,TMG_CfPeriForces);
    cusph::InteractionPeri_Shepard(TKernel,CellMode,bsshepard,PeriZone->GetListNp(),PeriZone->GetList(),PeriZone->GetListFluidIni(),PeriZone->GetListCellPart(),PeriZone->GetNc1(),PeriZone->GetBoxFluid(),PeriZone->GetBeginEndCell(),PeriZone->GetPosPres(),PosVolg,FdRhopg,FdWabg);
    TmgStop(Timers,TMG_CfPeriForces);
  }
}

//==============================================================================
/// Applies Shepard density filter.
//==============================================================================
void JSphGpuSingle::RunShepard(){
  TmgStart(Timers,TMG_CfShepard);
  //-Preparares Shepard interaction.
  PreInteraction_Shepard(!PeriActive);
  //-Shepard interaction.
  const unsigned bsshepard=BlockSizes.shepard;
  cusph::Interaction_Shepard(TKernel,CellMode,bsshepard,Npb,Np,CellDivSingle->GetNcells(),CellDivSingle->GetCellPart(),CellDivSingle->GetBeginCell(),PosVolg,FdRhopg,FdWabg);
  if(PeriActive){
    TmgStop(Timers,TMG_CfShepard);
    PeriInteractionShepard();
    cusph::Compute_Shepard(Npb,Np,MassFluid,FdRhopg,FdWabg,Rhopg);   //-Computes new values of density in Rhopg[].
    TmgStart(Timers,TMG_CfShepard);
  }
  else cudaMemcpy(Rhopg+Npb,FdRhopg+Npb,sizeof(float)*(Np-Npb),cudaMemcpyDeviceToDevice);//-Copies new values of density in Rhopg[].
  //-Releases the assigned memory.
  PosInteraction_Shepard();
  CheckCudaError("RunShepard","Failed while executing Shepard filter.");
  TmgStop(Timers,TMG_CfShepard);
}

//==============================================================================
/// Runs the simulation.
//==============================================================================
void JSphGpuSingle::Run(std::string appname,JCfgRun *cfg,JLog2 *log){
  const char* met="Run";
  if(!cfg||!log)return;
  AppName=appname; Log=log;

  //-Selection of GPU.
  //-------------------
  SelecDevice(cfg->GpuId);

  //-Configures timers.
  //-------------------
  TmgCreation(Timers,cfg->SvTimers);
  TmgStart(Timers,TMG_Init);

  //-Loads parameters and input data.
  //-----------------------------------------
  LoadConfig(cfg);
  LoadCaseParticles();
  ConfigConstants(Simulate2D);
  ConfigDomain();
  ConfigRunMode("Single-Gpu");

  //-Initialisation of variables of execution.
  //-------------------------------------------
  InitRun();
  UpdateMaxValues();
  PrintAllocMemory(GetAllocMemoryCpu(),GetAllocMemoryGpu());
  TmgStop(Timers,TMG_Init);
  SaveData(); 
  PartNstep=-1; Part++;
  //is it multiphase?
  if (Phases>1)Multiphase=true;

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
    //if(Nstep>=2)break;
  }
  TimerSim.Stop(); TimerTot.Stop();

  //-End of simulation.
  //--------------------
  FinishRun(partoutstop);
}

//==============================================================================
/// Generates ouput files of particle data.
//==============================================================================
void JSphGpuSingle::SaveData(){
  //v4 only
  const bool save=(SvData!=SDAT_None&&SvData!=SDAT_Info);
  const unsigned npsave=Np;//-NpbPer-NpfPer; //-Resta las periodicas si las hubiera. //-Subtracts periodic particles if any.
  //-Recovers particle data from GPU.
  if(save){
    TmgStart(Timers,TMG_SuDownData);
    ParticlesDataDown(Np,0,false,true);
    TmgStop(Timers,TMG_SuDownData);
  }

  //-Recupera datos de floatings en GPU.
  //-Retrieve floating object data from the GPU
  if(FtCount){
	//disabled
    TmgStart(Timers,TMG_SuDownData);
    //cudaMemcpy(FtoCenter,FtoCenterg,sizeof(double3)*FtCount,cudaMemcpyDeviceToHost);
    //for(unsigned cf=0;cf<FtCount;cf++)FtObjs[cf].center=FtoCenter[cf];
    //tfloat3 *aux=(tfloat3 *)FtoCenter;
    //cudaMemcpy(aux,FtoVelg,sizeof(float3)*FtCount,cudaMemcpyDeviceToHost);
    //for(unsigned cf=0;cf<FtCount;cf++)FtObjs[cf].fvel=aux[cf];
    //cudaMemcpy(aux,FtoOmegag,sizeof(float3)*FtCount,cudaMemcpyDeviceToHost);
    //for(unsigned cf=0;cf<FtCount;cf++)FtObjs[cf].fomega=aux[cf];
    TmgStop(Timers,TMG_SuDownData);
  }

  PartsOut->Clear();
  StInfoPartPlus infoplus;
  memset(&infoplus,0,sizeof(StInfoPartPlus));
  if(SvData&SDAT_Info){ 
    infoplus.nct=CellDivSingle->GetNct();
    infoplus.npbin=NpbOk;
    infoplus.npbout=Npb-NpbOk;
    infoplus.npf=Np-Npb;
    infoplus.npbper=0;//NpbPer;
    infoplus.npfper=0;//NpfPer;
    infoplus.memorycpualloc=this->GetAllocMemoryCpu();
    infoplus.gpudata=true;
    infoplus.memorynctalloc=infoplus.memorynctused=GetMemoryGpuNct();
    infoplus.memorynpalloc=infoplus.memorynpused=GetMemoryGpuNp();
    TimerSim.Stop();
    infoplus.timesim=TimerSim.GetElapsedTimeD()/1000.;
  }

  const tdouble3 vdom[2]={ToTDouble3(OrderDecode(CellDivSingle->GetDomainLimits(true))),ToTDouble3(OrderDecode(CellDivSingle->GetDomainLimits(false)))};
  JSph::SaveData(npsave,Idp,Pos,Vel,Rhop,Press,Viscop,Vtau,Idpm,1,vdom,&infoplus);


  TmgStop(Timers,TMG_SuSavePart);   
}

//==============================================================================
/// Displays and stores final brief of execution.
//==============================================================================
void JSphGpuSingle::FinishRun(bool stop){
  float tsim=TimerSim.GetElapsedTimeF()/1000.f,ttot=TimerTot.GetElapsedTimeF()/1000.f;
  JSph::ShowResume(stop,tsim,ttot,true,"");
  string hinfo=";RunMode",dinfo=string(";")+RunMode;
  if(SvTimers){
    ShowTimers();
    GetTimersInfo(hinfo,dinfo);
  }
  Log->Print(" ");
  if(SvRes)SaveRes(tsim,ttot,hinfo,dinfo);
}


