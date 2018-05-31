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

/// \file JSphGpu.cpp \brief Implements the class \ref JSphGpu.

#include "JSphGpu.h"
#include "JSphGpu_ker.h"

//#include "GMultiGpu_ker.h"

#include "Functions.h"
#include "JSphMotion.h"
#include "JPtxasInfo.h"
#include "JCellDivGpu.h"
#include "JPartsOut.h"
#include "JGpuArrays.h"
#include "JFormatFiles2.h"
#include "JSphDtFixed.h"
#include "JFloatingData.h"
#include "JSphVarAcc.h"

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JSphGpu::JSphGpu(){
  ClassName="JSphGpu";
  Cpu=false;
  Idp=NULL; Code=NULL; Pos=NULL; Vel=NULL; Rhop=NULL; //Velrhop=NULL;

  //Aditional
  Press=NULL; Viscop=NULL; Vtau=NULL;
  //Multi arrays
  Idpm=NULL;
 
  CellDiv=NULL;
  PartsOut=new JPartsOut;
  GpuArrays=new JGpuArrays;
  InitVars();
  TmgCreation(Timers,false);
}

//==============================================================================
/// Destructor.
//==============================================================================
JSphGpu::~JSphGpu(){
  FreeCpuMemoryParticles();
  FreeGpuMemoryParticles();
  delete GpuArrays;
  delete PartsOut;
  TmgDestruction(Timers);
  cudaDeviceReset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JSphGpu::InitVars(){
  RunMode="";
  Scell=0;
  Np=Npb=NpbOk=0;
  WithFloating=false;
  FreeCpuMemoryParticles();
  memset(&BlockSizes,0,sizeof(StBlockSizes));
  BlockSizesStr="";
  Idpg=NULL; Codeg=NULL; Posg=NULL; Velg=NULL; Rhopg=NULL;
  VelM1g=NULL; RhopM1g=NULL;                 //-Verlet.
  PosPreg=NULL; VelPreg=NULL; RhopPreg=NULL; //-Symplectic.
  Taug=NULL;  Csphg=NULL;                    //-Laminar+SPS.
  FdWabg=NULL;  FdRhopg=NULL; PosVolg=NULL;  //-Shepard.
  PosPresg=NULL; VelRhopg=NULL;
  ViscDtg=NULL; Arg=NULL; Aceg=NULL; VelXcorg=NULL; Deltag=NULL; 
  RidpMoving=NULL; 
  FtRidpg=NULL; FtDistg=NULL; FtOmegaVelg=NULL; FtoMasspg=NULL;

  //Additional
  Pressg=NULL; Viscopg=NULL; Vtaug=NULL; Surfg=NULL; Cderivg=NULL;
  //multi
  Idpmg=NULL; 
  FreeGpuMemoryParticles();
}

//==============================================================================
/// Throws exception due to a CUDA error.
//==============================================================================
void JSphGpu::RunExceptionCuda(const std::string &method,const std::string &msg,cudaError_t error){
  char cad[2048]; 
  sprintf(cad,"%s (CUDA error: %s).\n",msg.c_str(),cudaGetErrorString(error)); 
  Log->Print(GetExceptionText(method,cad));
  RunException(method,msg);
}

//==============================================================================
/// Checks error and throws exception.
//==============================================================================
void JSphGpu::CheckCudaError(const std::string &method,const std::string &msg){
  cudaError_t err=cudaGetLastError();
  if(err!=cudaSuccess)RunExceptionCuda(method,msg,err);
}

//==============================================================================
/// Releases memory in CPU of main data of particles.
//==============================================================================
void JSphGpu::FreeCpuMemoryParticles(){
  CpuParticlesSize=0;
  MemCpuParticles=0;
  delete[] Idp;   Idp=NULL;
  delete[] Code;  Code=NULL;
  delete[] Pos;   Pos=NULL;
  delete[] Vel;   Vel=NULL;
  delete[] Rhop;  Rhop=NULL;

  //delete[] Velrhop;    Velrhop=NULL;

  //adtitional
  delete[] Press;	Press=NULL;
  delete[] Viscop;  Viscop=NULL;
  delete[] Vtau;	Vtau=NULL;
  delete[] Idpm;	Idpm=NULL;
  
}

//==============================================================================
/// Allocates memory in CPU of main data of particles.
//==============================================================================
void JSphGpu::AllocCpuMemoryParticles(unsigned np){
  const char* met="AllocCpuMemoryParticles";
  FreeCpuMemoryParticles();
  CpuParticlesSize=np;
  if(np>0){
    try{
      Idp=new unsigned[np];   MemCpuParticles+=sizeof(unsigned)*np;
      Code=new word[np];      MemCpuParticles+=sizeof(word)*np;
      Pos=new tfloat3[np];    MemCpuParticles+=sizeof(tfloat3)*np;
      Vel=new tfloat3[np];    MemCpuParticles+=sizeof(tfloat3)*np;
      Rhop=new float[np];     MemCpuParticles+=sizeof(float)*np;

	  //Velrhop=new tfloat4[np];   MemCpuParticles+=sizeof(tfloat4)*np;

	  //Addtitional
	  Press=new float[np];			MemCpuParticles+=sizeof(float)*np;	
	  Viscop=new float[np];			MemCpuParticles+=sizeof(float)*np;	
	  Vtau=new tsymatrix3f[np];		MemCpuParticles+=sizeof(tsymatrix3f)*np;
	  Idpm=new unsigned[np];		MemCpuParticles+=sizeof(unsigned)*np;	
    }
    catch(const std::bad_alloc){
      char cad[128]; sprintf(cad,"The requested memory could not be allocated (np=%u).",np);
      RunException(met,cad);
    }
  }
}

//==============================================================================
/// Releases memory in GPU of main data of particles.
//==============================================================================
void JSphGpu::FreeGpuMemoryParticles(){
  GpuParticlesSize=0;
  MemGpuParticles=0;
  GpuArrays->Reset();
  if(RidpMoving)cudaFree(RidpMoving);    RidpMoving=NULL;
  if(FtRidpg)cudaFree(FtRidpg);          FtRidpg=NULL;
  if(FtDistg)cudaFree(FtDistg);          FtDistg=NULL;
  if(FtOmegaVelg)cudaFree(FtOmegaVelg);  FtOmegaVelg=NULL;
  if(FtoMasspg)cudaFree(FtoMasspg);      FtoMasspg=NULL;
}

//==============================================================================
/// Allocates memory in GPU of main data of particles.
//==============================================================================
void JSphGpu::AllocGpuMemoryParticles(unsigned np,float over){
  const char* met="AllocGpuMemoryParticles";
  FreeGpuMemoryParticles();
  //-Computes the number of particles for which memory will be allocated.
  const unsigned np2=(over>0? unsigned(over*np): np);
  GpuParticlesSize=np2;
  //-Computes how many arrays.
  GpuArrays->SetArraySize(np2);
  GpuArrays->AddArrayCount(JGpuArrays::SIZE_2B,2);    //-code,code2.
  GpuArrays->AddArrayCount(JGpuArrays::SIZE_4B,4);    //-idp,rhop,ar,viscdt.
  if(TDeltaSph==DELTA_DBCExt)GpuArrays->AddArrayCount(JGpuArrays::SIZE_4B,1);  //-delta.
  GpuArrays->AddArrayCount(JGpuArrays::SIZE_12B,4);   //-pos,vel,ace,velxcor.
  //Additional
  GpuArrays->AddArrayCount(JGpuArrays::SIZE_4B,8);    //-Idpm, Press, Viscop, CV, surf x2 for reordering
  GpuArrays->AddArrayCount(JGpuArrays::SIZE_4B,1);	  //?? dont know just one more for the multiphase debug
  GpuArrays->AddArrayCount(JGpuArrays::SIZE_12B,2);	  //Cderiv

  if(TStep==STEP_Verlet){
    GpuArrays->AddArrayCount(JGpuArrays::SIZE_4B,1);  //-rhopm1.
    GpuArrays->AddArrayCount(JGpuArrays::SIZE_12B,1); //-velm1.
  }
  else if(TStep==STEP_Symplectic){
    GpuArrays->AddArrayCount(JGpuArrays::SIZE_4B,1);  //-rhoppre.
    GpuArrays->AddArrayCount(JGpuArrays::SIZE_12B,2); //-pospre,velpre.
  }
  GpuArrays->AddArrayCount(JGpuArrays::SIZE_16B,2);   //-pospres,velrhop.
  if(TVisco==VISCO_LaminarSPS){
    GpuArrays->AddArrayCount(JGpuArrays::SIZE_24B,2); //-tau,csph.
  }
  //extra formulation
  if(TVisco==VISCO_SumSPS){
    GpuArrays->AddArrayCount(JGpuArrays::SIZE_24B,5); //-vtau,tau,csph. 
  }
  // I know I dont need all this memory will reuce it soon

  //-Allocates memory for arrays with fixed size.
  long long size=0;
  if(CaseNmoving){
    size_t m=sizeof(unsigned)*CaseNmoving;
    cudaMalloc((void**)&RidpMoving,m); size+=m;
  }
  if(CaseNfloat){
    size_t m=sizeof(unsigned)*CaseNfloat;
    cudaMalloc((void**)&FtRidpg,m);     size+=m;
    m=sizeof(float3)*CaseNfloat;
    cudaMalloc((void**)&FtDistg,m);     size+=m;
    cudaMalloc((void**)&FtOmegaVelg,m); size+=m;
    m=sizeof(float)*FtCount;
    cudaMalloc((void**)&FtoMasspg,m);     size+=m;
  }
  //-Shows the allocated memory.
  MemGpuParticles=GpuArrays->GetAllocMemoryGpu()+size;
  PrintSizeNp(np2,MemGpuParticles);
  CheckCudaError(met,"GPU memory allocation failed.");
}

//==============================================================================
/// Arrays for basic data of particles. 
//==============================================================================
void JSphGpu::ReserveBasicGpuArrays(){
  Idpg=GpuArrays->ReserveUint();
  Codeg=GpuArrays->ReserveWord();
  Rhopg=GpuArrays->ReserveFloat();
  Posg=GpuArrays->ReserveFloat3();
  Velg=GpuArrays->ReserveFloat3();
  //Aditional
  Pressg=GpuArrays->ReserveFloat();
  Viscopg=GpuArrays->ReserveFloat();
  //CVg=GpuArrays->ReserveFloat();  //->debug only
  //Surfg=GpuArrays->ReserveFloat();
  //Cderivg=GpuArrays->ReserveFloat3();
  Idpmg=GpuArrays->ReserveUint();

  if(TStep==STEP_Verlet){
    RhopM1g=GpuArrays->ReserveFloat();
    VelM1g=GpuArrays->ReserveFloat3();
  }
  if(TVisco==VISCO_LaminarSPS){
    Taug=GpuArrays->ReserveSymatrix3f();
  }
  if(TVisco==VISCO_SumSPS){
    Taug=GpuArrays->ReserveSymatrix3f();
	Vtaug=GpuArrays->ReserveSymatrix3f();
  }
}

//==============================================================================
/// Returns the allocated memory in CPU.
//==============================================================================
long long JSphGpu::GetAllocMemoryCpu()const{  
  long long s=JSph::GetAllocMemoryCpu();
  //Allocated in AllocMemoryParticles().
  s+=MemCpuParticles;
  //Allocated in other objects.
  if(PartsOut)s+=PartsOut->GetAllocMemory();
  return(s);
}

//==============================================================================
/// Returns the allocated memory in CPU.
//==============================================================================
long long JSphGpu::GetAllocMemoryGpu()const{  
  long long s=0;
  //Allocated in GpuAllocMemoryParticles().
  s+=MemGpuParticles;
  //Allocated in other objects.
  return(s);
}

//==============================================================================
/// Displays the allocated memory.
//==============================================================================
void JSphGpu::PrintAllocMemory(long long mcpu,long long mgpu)const{
  char cad[128];
  sprintf(cad,"Allocated memory in CPU: %lld (%.2f MB)",mcpu,double(mcpu)/(1024*1024)); Log->Print(cad);
  sprintf(cad,"Allocated memory in GPU: %lld (%.2f MB)",mgpu,double(mgpu)/(1024*1024)); Log->Print(cad);
}

//==============================================================================
/// Uploads particle data to GPU.
//==============================================================================
void JSphGpu::ParticlesDataUp(unsigned n){
  cudaMemcpy(Idpg,Idp,sizeof(unsigned)*n,cudaMemcpyHostToDevice);
  cudaMemcpy(Codeg,Code,sizeof(word)*n,cudaMemcpyHostToDevice);
  cudaMemcpy(Posg,Pos,sizeof(float3)*n,cudaMemcpyHostToDevice);
  cudaMemcpy(Velg,Vel,sizeof(float3)*n,cudaMemcpyHostToDevice);
  cudaMemcpy(Rhopg,Rhop,sizeof(float)*n,cudaMemcpyHostToDevice);
  //Aditional
  cudaMemcpy(Pressg,Press,sizeof(float)*n,cudaMemcpyHostToDevice);
  cudaMemcpy(Viscopg,Viscop,sizeof(float)*n,cudaMemcpyHostToDevice);
  if(TVisco==VISCO_SumSPS)cudaMemcpy(Vtaug,Vtau,sizeof(tsymatrix3f)*n,cudaMemcpyHostToDevice);
  //cvg,surf,cderiv no need to copy *(except if you wana restart the sim etc)
  //multi
  cudaMemcpy(Idpmg,Idpm,sizeof(unsigned)*n,cudaMemcpyHostToDevice);
  
  CheckCudaError("ParticlesDataUp","Failed copying data to GPU.");
}

//==============================================================================
/// Recovers particle data from GPU.
//==============================================================================
void JSphGpu::ParticlesDataDown(unsigned n,unsigned pini,bool code,bool orderdecode){
  cudaMemcpy(Idp,Idpg+pini,sizeof(unsigned)*n,cudaMemcpyDeviceToHost);
  cudaMemcpy(Pos,Posg+pini,sizeof(float3)*n,cudaMemcpyDeviceToHost);
  cudaMemcpy(Vel,Velg+pini,sizeof(float3)*n,cudaMemcpyDeviceToHost);
  cudaMemcpy(Rhop,Rhopg+pini,sizeof(float)*n,cudaMemcpyDeviceToHost);
  CheckCudaError("ParticlesDataDown","Failed copying data from GPU.");
  //Additional
  
  cudaMemcpy(Press,Pressg+pini,sizeof(float)*n,cudaMemcpyDeviceToHost);
  cudaMemcpy(Viscop,Viscopg+pini,sizeof(float)*n,cudaMemcpyDeviceToHost);
  cudaMemcpy(Idpm,Idpmg+pini,sizeof(unsigned)*n,cudaMemcpyDeviceToHost);
  CheckCudaError("ParticlesDataDown","Failed copying data from GPU.");
  if(TVisco==VISCO_SumSPS)cudaMemcpy(Vtau,Vtaug+pini,sizeof(tsymatrix3f)*n,cudaMemcpyDeviceToHost); //For debug stress visuallisation only
  CheckCudaError("ParticlesDataDown","Failed copying data from GPU.");
  //printf("%f\t%f\t%f",Viscop[100],Viscop[1500],Viscop[2800]);
  //getchar();

  if(code)cudaMemcpy(Code,Codeg+pini,sizeof(word)*n,cudaMemcpyDeviceToHost);
  if(orderdecode&&CellOrder!=ORDER_XYZ){
    OrderDecodeData(CellOrder,n,Pos);
    OrderDecodeData(CellOrder,n,Vel);
  }
  CheckCudaError("ParticlesDataDown","Failed copying data from GPU.");
}

//==============================================================================
/// Initialises CUDA device. 
//==============================================================================
void JSphGpu::SelecDevice(int gpuid){
  const char* met="SelecDevice";
  char cad[1024];
  Log->Print("[Select CUDA Device]");
  GpuSelect=-1;
  int devcount;
  cudaGetDeviceCount(&devcount);
  CheckCudaError(met,"Failed getting devices info.");
  for(int dev=0;dev<devcount;dev++){
    cudaDeviceProp devp;
    cudaGetDeviceProperties(&devp,dev);
    sprintf(cad,"Device %d: \"%s\"",dev,devp.name); Log->Print(cad);
    sprintf(cad,"  Compute capability:         %d.%d",devp.major,devp.minor); Log->Print(cad);
    int corebymp=(devp.major==1? 8: (devp.major==2? (devp.minor==0? 32: 48): (devp.major==3? 192: -1)));
    sprintf(cad,"  Multiprocessors:           %d (%d cores)",devp.multiProcessorCount,devp.multiProcessorCount*corebymp); Log->Print(cad);
    sprintf(cad,"  Memory global:             %d MB",int(devp.totalGlobalMem/(1024*1024))); Log->Print(cad);
    sprintf(cad,"  Clock rate:                %.2f GHz",devp.clockRate*1e-6f); Log->Print(cad);
    #if CUDART_VERSION >= 2020
    sprintf(cad,"  Run time limit on kernels: %s",(devp.kernelExecTimeoutEnabled? "Yes": "No")); Log->Print(cad);
    #endif
    #if CUDART_VERSION >= 3010
    sprintf(cad,"  ECC support enabled:       %s",(devp.ECCEnabled? "Yes": "No")); Log->Print(cad);
    #endif
  }
  Log->Print("");
  if(devcount){
    if(gpuid>=0)cudaSetDevice(gpuid);
    else{
      unsigned *ptr=NULL;
      cudaMalloc((void**)&ptr,sizeof(unsigned)*100);
      cudaFree(ptr);
    }
    cudaDeviceProp devp;
    int dev;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&devp,dev);
    GpuSelect=dev;
    GpuName=devp.name;
    GpuGlobalMem=devp.totalGlobalMem;
    GpuSharedMem=int(devp.sharedMemPerBlock);
    GpuCompute=devp.major*10+devp.minor;
    //-Shows information of the selected hardware.
    Log->Print("[GPU Hardware]");
    if(gpuid<0)sprintf(cad,"Gpu_%d?=\"%s\"",GpuSelect,GpuName.c_str());
    else sprintf(cad,"Gpu_%d=\"%s\"",GpuSelect,GpuName.c_str());
    Hardware=cad;
    if(gpuid<0)sprintf(cad,"Device default: %d  \"%s\"",GpuSelect,GpuName.c_str());
    else sprintf(cad,"Device selected: %d  \"%s\"",GpuSelect,GpuName.c_str()); Log->Print(cad);
    sprintf(cad,"Compute capability: %.1f",float(GpuCompute)/10); Log->Print(cad);
    sprintf(cad,"Memory global: %d MB",int(GpuGlobalMem/(1024*1024))); Log->Print(cad);
    sprintf(cad,"Memory shared: %u Bytes",GpuSharedMem); Log->Print(cad);
  }
  else RunException(met,"There are no available CUDA devices.");
}

//==============================================================================
/// Returns the optimum size of block according to number of registers of the CUDA kernel and compute capabality.
//==============================================================================
unsigned JSphGpu::OptimizeBlockSize(unsigned compute,unsigned nreg){
  //printf("compute:%d\n",compute);
  if(compute>=30){
    if(nreg<=32)return(256);       // 1-32 -> 128:100%  256:100%  512:100%
    else if(nreg<=40)return(256);  //33-40 -> 128:75%  256:75%  384:75%  512:75%
    else if(nreg<=48)return(256);  //41-48 -> 128:63%  256:63%
    else if(nreg<=56)return(128);  //49-56 -> 128:56%  256:50%  384:56%
    else if(nreg<=63)return(256);  //49-63 -> 128:50%  256:50%  512:50%
    else return(256);              
  }
  else if(compute>=20){
    if(nreg<=20)return(256);       // 1-20 -> 192:100%  256:100%  384:100%
    else if(nreg<=24)return(224);  //21-24 -> 192:88%  224:88%  256:83%  448:88%
    else if(nreg<=26)return(192);  //25-26 -> 192:75%  256:67%  288:75%  416:81%
    else if(nreg<=28)return(192);  //27-28 -> 192:75%  256:67%  288:75%
    else if(nreg<=32)return(256);  //29-32 -> 256:67%
    else if(nreg<=34)return(192);  //33-34 -> 192:63%  256:50%
    else if(nreg<=36)return(128);  //35-36 -> 128:58%  224:58%  256:50%
    else if(nreg<=42)return(128);  //37-42 -> 128:50%  256:50%
    else if(nreg<=44)return(224);  //43-44 -> 192:38%  224:44%  256:33%  352:46%
    else if(nreg<=50)return(128);  //45-50 -> 128:42%  160:42%  256:33%  320:42%
    else if(nreg<=56)return(192);  //51-56 -> 192:38%  256:33%  288:38%
    else if(nreg<=63)return(128);  //57-63 -> 128:33%  256:33%
    else return(128);              
  }
  else if(compute>=12){
    if(nreg<=16)return(256);       // 1-16 -> 128:100%  256:100%
    else if(nreg<=18)return(448);  //17-18 -> 128:75%  192:75%  256:75%  448:88%
    else if(nreg<=20)return(256);  //19-20 -> 192:75%  256:75%  384:75%
    else if(nreg<=21)return(192);  //21    -> 192:75%  256:50%  288:56%  320:63%  352:69%  384:75%
    else if(nreg<=24)return(128);  //22-24 -> 128:63%  192:56%  288:56%  256:50%  320:63%
    else if(nreg<=25)return(320);  //25    -> 192:56%  288:56%  256:50%  320:63%
    else if(nreg<=26)return(192);  //26    -> 192:56%  256:50%
    else if(nreg<=32)return(256);  //27-32 -> 256:50%
    else if(nreg<=36)return(448);  //33-36 -> 192:38%  256:25%  416:41%  448:44%
    else if(nreg<=42)return(192);  //37-42 -> 192:38%  256:25%
    else if(nreg<=51)return(320);  //43-51 -> 256:25%  288:28%  320:31%
    else if(nreg<=64)return(256);  //52-64 -> 128:25%  256:25%
    else return(192);              //65-85 -> 128:13%  192:19%
  }
  else if(compute>=10){
    if(nreg<=10)return(256);       // 1-10 -> 128:100%  192:100%  256:100%  384:100%
    else if(nreg<=12)return(128);  //11-12 -> 128:83%  192:75%  256:67%  320:83%
    else if(nreg<=13)return(192);  //13    -> 128:67%  192:75%  256:67%
    else if(nreg<=16)return(256);  //14-16 -> 128:67%  192:50%  256:67%
    else if(nreg<=18)return(448);  //17-18 -> 128:50%  192:50%  256:33%  384:50%  448:58%
    else if(nreg<=20)return(128);  //19-20 -> 128:50%  192:50%  256:33%  384:50%
    else if(nreg<=21)return(192);  //21    -> 128:33%  192:50%  256:33%  384:50%
    else if(nreg<=24)return(320);  //22-24 -> 64:42%  128:33%  256:33%  320:42%
    else if(nreg<=25)return(320);  //25    -> 64:33%  128:33%  256:33%  320:42%
    else if(nreg<=32)return(256);  //26-32 -> 64:33%  128:33%  256:33%
    else if(nreg<=40)return(192);  //33-40 -> 64:25%  128:17%  192:25%
    else if(nreg<=42)return(192);  //41-42 -> 64:17%  128:17%  192:25%
    else if(nreg<=64)return(128);  //43-64 -> 64:17%  128:17%
    else return(64);               //65-128-> 64:8%
  }
  return(256);
}

//==============================================================================
/// Returns BlockSize as function of registers of the CUDA kernel.
//==============================================================================
unsigned JSphGpu::BlockSizeConfig(const string& opname,unsigned compute,unsigned regs){
  char cad[1024];
  unsigned bsize=256;
  if(regs){
    bsize=OptimizeBlockSize(compute,regs);
    sprintf(cad,"%s=%u (%u regs)",opname.c_str(),bsize,regs);
  }
  else sprintf(cad,"%s=%u (? regs)",opname.c_str(),bsize);
  Log->Print(cad);
  if(!BlockSizesStr.empty())BlockSizesStr=BlockSizesStr+", ";
  BlockSizesStr=BlockSizesStr+cad;
  return(bsize);
}

//==============================================================================
/// Configures data of DeviceContext & DeviceCtes. Returns true in case of error.
//==============================================================================
void JSphGpu::ConfigBlockSizes(bool usezone,bool useperi){
  const char met[]="ConfigBlockSizes";
  //-Gets configuration according to CellMode.
  //--------------------------------------
  Log->Print(" ");
  Log->Print(fun::VarStr("PtxasFile",PtxasFile));
  const unsigned smgpu=(GpuCompute<30? (GpuCompute<20? (GpuCompute<12? 10: 12): 20): 30);
  unsigned smcode=(smgpu==13? 12: smgpu);
  JPtxasInfo pt;
  if(fun::FileExists(PtxasFile)){
    pt.LoadFile(PtxasFile);
    if(smgpu==20&&!pt.CheckSm(20))RunException(met,"Code is not compiled for sm20.");
    if(smgpu==30&&!pt.CheckSm(30)){
      if(!pt.CheckSm(20))RunException(met,"Code is not compiled for sm20 and sm30.");
      else smcode=20;
    }
    sprintf(Cad,"Using code for compute capability %3.1f on hardware %3.1f",float(smcode)/10,float(smgpu)/10); Log->Print(Cad);
  }
  else Log->Print("**Without optimization of registers.");
  pt.SaveCsv(DirOut+"ptxas_info.csv");
  BlockSizesStr="";
  if(CellMode==CELLMODE_2H||CellMode==CELLMODE_H){
    bool floating=(CaseNfloat>0);
    const unsigned hdivfc=Hdiv;
    string kerbound=(floating? "cusph_KerFtInteractionForcesBound": "cusph_KerInteractionForcesBound");
    string kerfluid=(floating? "cusph_KerFtInteractionForcesFluid": "cusph_KerInteractionForcesFluid");
    BlockSizes.forcesbound=BlockSizeConfig("BsForcesBound",smgpu,pt.GetRegs(kerbound,smcode,TKernel,hdivfc));
    BlockSizes.forcesfluid=BlockSizeConfig("BsForcesFluid",smgpu,pt.GetRegs(kerfluid,smcode,TDeltaSph,TKernel,TVisco,true,hdivfc));
    if(TStep==STEP_Symplectic)BlockSizes.forcesfluidcorr=BlockSizeConfig("BsForcesFluidCorr",smcode,pt.GetRegs(kerfluid,smcode,TDeltaSph,TKernel,TVisco,false,hdivfc));
    if(ShepardSteps)BlockSizes.shepard=BlockSizeConfig("BsShepard",smgpu,pt.GetRegs("cusph_KerInteractionShepard",smcode,TKernel,Hdiv));
    if(useperi){
      string txzone="Periodic";
      BlockSizes.forcesbound_peri=BlockSizeConfig(string("BsForcesBound")+txzone,smgpu,pt.GetRegs("cusph_KerInteractionPeriForcesBound",smcode,TKernel,Hdiv));
      BlockSizes.forcesfluid_peri=BlockSizeConfig(string("BsForcesFluid")+txzone,smgpu,pt.GetRegs("cusph_KerInteractionPeriForcesFluid",smcode,TDeltaSph,TKernel,TVisco,true,Hdiv));
      if(TStep==STEP_Symplectic)BlockSizes.forcesfluidcorr_peri=BlockSizeConfig(string("BsForcesFluidCorr")+txzone,smgpu,pt.GetRegs("cusph_KerInteractionPeriForcesFluid",smcode,TDeltaSph,TKernel,TVisco,false,Hdiv));
      if(ShepardSteps)BlockSizes.shepard_peri=BlockSizeConfig(string("BsShepard")+txzone,smgpu,pt.GetRegs("cusph_KerInteractionPeriShepard",smcode,TKernel,Hdiv));
    }
  }
  else RunException(met,"CellMode unrecognised.");
  Log->Print(" ");
}

//==============================================================================
/// Configures mode of GPU execution.
//==============================================================================
void JSphGpu::ConfigRunMode(const std::string &preinfo){
  RunMode=preinfo+RunMode;
  if(Stable)RunMode=string("Stable, ")+RunMode;
  Log->Print(fun::VarStr("RunMode",RunMode));
  if(!RunMode.empty())RunMode=RunMode+", "+BlockSizesStr;
}

//==============================================================================
/// Adjusts variables of particles of floating bodies.
//==============================================================================
void JSphGpu::InitFloating(){
  JSph::InitFloatingData();
  if(!PartBegin){
    //-Gets positions of floating particles.
    cudaMemset(FtRidpg,255,sizeof(unsigned)*CaseNfloat); //-Assigns UINT_MAX values.
    cusph::CalcRidp(Np-Npb,Npb,CaseNpb,CaseNpb+CaseNfloat,Idpg,FtRidpg);
    //-Computes distance of particles to center of the object.
    for(unsigned cf=0;cf<FtCount;cf++){
      StFloatingData *fobj=FtObjs+cf;
      cusph::FtCalcDist(fobj->count,fobj->begin-CaseNpb,fobj->center,FtRidpg,Posg,FtDistg);
    }
    //-Stores distance of particles to center of the object.
    tfloat3 *dist=new tfloat3[CaseNfloat];
    cudaMemcpy(dist,FtDistg,sizeof(tfloat3)*CaseNfloat,cudaMemcpyDeviceToHost);
    if(CellOrder!=ORDER_XYZ)OrderDecodeData(CellOrder,CaseNfloat,dist);
    FtData->AddDist(CaseNfloat,dist);
    delete[] dist;
  }
  else{
    //-Recovers distance of particles to center of the object.
    tfloat3 *dist=new tfloat3[CaseNfloat];
    memcpy(dist,FtData->GetDist(CaseNfloat),sizeof(tfloat3)*CaseNfloat);
    if(CellOrder!=ORDER_XYZ)OrderCodeData(CellOrder,CaseNfloat,dist);
    cudaMemcpy(FtDistg,dist,sizeof(tfloat3)*CaseNfloat,cudaMemcpyHostToDevice);
    delete[] dist;
  }
  //-Copys massp values to GPU
  for(unsigned cf=0;cf<FtCount;cf++)cudaMemcpy(FtoMasspg+cf,&(FtObjs[cf].massp),sizeof(float),cudaMemcpyHostToDevice);
}

//==============================================================================
/// Initialises arrays and variables for the GPU execution.
//==============================================================================
void JSphGpu::InitRun(){
  if(TStep==STEP_Verlet){
    cudaMemcpy(VelM1g,Velg,sizeof(tfloat3)*Np,cudaMemcpyDeviceToDevice);
    cudaMemcpy(RhopM1g,Rhopg,sizeof(float)*Np,cudaMemcpyDeviceToDevice);
    VerletStep=0;
  }
  else if(TStep==STEP_Symplectic){
    DtPre=DtIni;
  }
  if(TVisco==VISCO_LaminarSPS){  
    cudaMemset(Taug,0,sizeof(tsymatrix3f)*Np);
  }
  

  //Additional
  if(TVisco==VISCO_SumSPS){  
	cudaMemset(Vtaug,0,sizeof(tsymatrix3f)*Np);
	cudaMemset(Taug,0,sizeof(tsymatrix3f)*Np);
  }
  
  //cudaMemset(CVg,0,sizeof(float)*Np); //->debug only
  //cudaMemset(Surfg,0,sizeof(float)*Np);
  //cudaMemset(Cderivg,0,sizeof(float3)*Np);
  
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

  
  //-Copies constants to GPU multiphase
  //Multiphase
  //PhaseCte and Phasearray copy to *g before copy to GPU memory
  StPhaseCteg* phasecte=new StPhaseCteg[Phases];
  StPhaseArrayg* phasearray=new StPhaseArrayg[Phases];
  memcpy(phasecte,PhaseCte,sizeof(StPhaseCteg)*Phases);
  memcpy(phasearray,PhaseArray,sizeof(StPhaseArrayg)*Phases);

  //-Copies constants to GPU Single phase general constants
  StCteInteraction ctes;
  ctes.nbound=CaseNbound;
  ctes.massb=MassBound; ctes.massf=MassFluid;
  ctes.fourh2=Fourh2; ctes.h=H;
  ctes.cubic_a1=CubicCte.a1; ctes.cubic_a2=CubicCte.a2; ctes.cubic_aa=CubicCte.aa; ctes.cubic_a24=CubicCte.a24;
  ctes.cubic_c1=CubicCte.c1; ctes.cubic_c2=CubicCte.c2; ctes.cubic_d1=CubicCte.d1; ctes.cubic_odwdeltap=CubicCte.od_wdeltap;
  ctes.wendland_awen=WendlandCte.awen; ctes.wendland_bwen=WendlandCte.bwen;
  ctes.cteshepard=CteShepard;
  ctes.cs0=Cs0; ctes.visco=Visco; ctes.eta2=Eta2; ctes.overrhop0=OverRhop0;
  ctes.delta2h=Delta2H;
  ctes.posminx=MapPosMin.x; ctes.posminy=MapPosMin.y; ctes.posminz=MapPosMin.z; ctes.scell=Scell; ctes.dosh=Dosh;
  //Additional  (I dont really use all that but let them in for now)
  Log->Print("\nLoading multiphase constants to GPU memory");
  ctes.phasecount=Phases;
  ctes.spssmag=SpsSmag; ctes.spsblin=SpsBlin;
  float min_visc=FLT_MAX;
  for (unsigned c=0;c<Phases;c++)min_visc=min(min_visc,PhaseCte[c].visco);
  ctes.minvisc=min_visc;

  cusph::CteInteractionUp(&ctes,phasecte,phasearray,Phases);
  CheckCudaError("InitRun","Failed copying constants to GPU.");
  Log->Print("Finish loading multiphase constants to GPU memory\n");

}

//==============================================================================
/// Adds variable acceleration from input files.
//==============================================================================
void JSphGpu::AddVarAcc(){
  for(unsigned c=0;c<VarAcc->GetCount();c++){
    unsigned mkvalue;
    tfloat3 acclin,accang,centre;
    VarAcc->GetAccValues(c,TimeStep,mkvalue,acclin,accang,centre);
    const word codesel=word(CODE_TYPE_FLUID|mkvalue);
    //sprintf(Cad,"t:%f  cod:%u  acclin.x:%f",TimeStep,codesel,acclin.x); Log->PrintDbg(Cad);
    cusph::AddVarAcc(Np-Npb,Npb,codesel,acclin,accang,centre,Codeg,Posg,Aceg);
  }
}

//==============================================================================
/// Prepares variables for interaction "INTER_Forces" or "INTER_ForcesCorr".
//==============================================================================
void JSphGpu::PreInteractionVars_Forces(TpInter tinter,unsigned ini,unsigned np,unsigned npb){
  //-Initialises arrays.
  cudaMemset(ViscDtg+ini,0,sizeof(float)*np);    //ViscDtg[]=0.
  cudaMemset(Arg+ini,0,sizeof(float)*np);        //Arg[]=0.
  if(Deltag)cudaMemset(Deltag+ini,0,sizeof(float)*np);                                   //Deltag[]=0.
  if(VelXcorg)cudaMemset(VelXcorg+ini+npb,0,sizeof(tfloat3)*(np-npb));                   //VelXcorg[]=(0,0,0).
  //stess
  if(TVisco==VISCO_LaminarSPS)cudaMemset(Csphg+ini+npb,0,sizeof(tsymatrix3f)*(np-npb));  //Csphg[]=(0,0,0,0,0,0).
  if(TVisco==VISCO_SumSPS)cudaMemset(Csphg+ini,0,sizeof(tsymatrix3f)*(np));
  if(TVisco==VISCO_SumSPS)cudaMemset(Viscopg+ini,0,sizeof(float)*(np));
  //multi delete them here all of them, leave only what you need for Jose, remov. shift
  cudaMemset(CVg+ini,2,sizeof(float)*(np));  //big number i.e. 2 cv values 0-1
  //shift
  //cudaMemset(Surfg+ini,0,sizeof(float)*(np));
  //cudaMemset(Cderivg+ini,0,sizeof(tfloat3)*(np));
 
  //Calculate Pressures
  cusph::PreInteraction_Forces(np,npb,Posg+ini,Velg+ini,Rhopg+ini,PosPresg+ini,VelRhopg+ini,Aceg+ini,Gravity,Pressg+ini,Idpmg+ini); //Added OverRhop0 to function call.
  //Calculate variable Tauy
  cusph::PreYieldStress(np,npb,Rhopg+ini,PosPresg+ini,Idpmg+ini,Viscopg);
  if(VarAcc)AddVarAcc();
  CheckCudaError("PreInteractionVars_Forces","Failed preparing vars for interaction.");
}

//==============================================================================
/// Prepares variables for interaction "INTER_Forces" or "INTER_ForcesCorr".
//==============================================================================
void JSphGpu::PreInteraction_Forces(TpInter tinter){
  TmgStart(Timers,TMG_CfPreForces);
  //-Assigns memory.
  Arg=GpuArrays->ReserveFloat();
  Aceg=GpuArrays->ReserveFloat3();
  VelXcorg=(tinter==INTER_Forces? GpuArrays->ReserveFloat3(): NULL);
  ViscDtg=GpuArrays->ReserveFloat();
  PosPresg=GpuArrays->ReserveFloat4();
  VelRhopg=GpuArrays->ReserveFloat4();
  if(TDeltaSph==DELTA_DBCExt)Deltag=GpuArrays->ReserveFloat();
  //csph
  if(TVisco==VISCO_LaminarSPS)Csphg=GpuArrays->ReserveSymatrix3f();
  if(TVisco==VISCO_SumSPS)Csphg=GpuArrays->ReserveSymatrix3f();
  //multi
  CVg=GpuArrays->ReserveFloat();
  //Surfg=GpuArrays->ReserveFloat();
  //Cderivg=GpuArrays->ReserveFloat3();
  
  
  //-Initialises arrays.
  PreInteractionVars_Forces(tinter,0,Np,Npb);
  //-Computes CsoundMax
  CsoundMax=Cs0;
  //Zero viscdt variables
  ViscDtMax=0;
  ForceDtMax=0;
  ViscoDtMax=0;
  TmgStop(Timers,TMG_CfPreForces);
}

//==============================================================================
/// Releases assigned memory of GpuArrays used in "INTER_Forces" or "INTER_ForcesCorr".
//==============================================================================
void JSphGpu::PosInteraction_Forces(){
  //-Releases memory assigned in PreInteraction_Forces().
  GpuArrays->Free(Arg);       Arg=NULL;
  GpuArrays->Free(Aceg);      Aceg=NULL;
  GpuArrays->Free(VelXcorg);  VelXcorg=NULL;
  GpuArrays->Free(ViscDtg);   ViscDtg=NULL;
  GpuArrays->Free(Deltag);    Deltag=NULL;
  GpuArrays->Free(PosPresg);  PosPresg=NULL;
  GpuArrays->Free(VelRhopg);  VelRhopg=NULL;
  GpuArrays->Free(Csphg);     Csphg=NULL;
  
  //Additional (commnet out for output as appropriate)
  GpuArrays->Free(CVg);			CVg=NULL;
  GpuArrays->Free(Surfg);		Surfg=NULL;
  GpuArrays->Free(Cderivg);     Cderivg=NULL;
 
  
}

//==============================================================================
/// Prepares variables for interaction "INTER_Shepard".
//==============================================================================
void JSphGpu::PreInteractionVars_Shepard(unsigned pini,unsigned pfin){
  float pmax=max(MapPosMax.x,max(MapPosMax.y,MapPosMax.z));
  float pmin=min(MapPosMin.x,min(MapPosMin.y,MapPosMin.z));
  float ftposout=pmin-(pmax-pmin+Dosh)*2;
  cusph::PreInteraction_Shepard(WithFloating,pini,pfin,Posg,Rhopg,Idpg,CaseNbound,ftposout,MassFluid,PosVolg);
}

//==============================================================================
/// Prepares variables for interaction "INTER_Shepard".
//==============================================================================
void JSphGpu::PreInteraction_Shepard(bool onestep){
  const unsigned npf=Np-Npb;
  FdRhopg=GpuArrays->ReserveFloat();
  FdWabg=NULL; if(!onestep)FdWabg=GpuArrays->ReserveFloat();
  PosVolg=GpuArrays->ReserveFloat4();
  PreInteractionVars_Shepard(Npb,Np);
}

//==============================================================================
/// Releases assigned memory of GpuArrays used in "INTER_Shepard".
//==============================================================================
void JSphGpu::PosInteraction_Shepard(){
  //-Releases memory assigned in PreInteraction_Shepard().
  GpuArrays->Free(FdRhopg);  FdRhopg=NULL;
  GpuArrays->Free(FdWabg);   FdWabg=NULL;
  GpuArrays->Free(PosVolg);  PosVolg=NULL;
}

//==============================================================================
/// Updates particles according to forces and dt using VERLET. 
//==============================================================================
void JSphGpu::ComputeVerlet(bool rhopbound,float dt){
  TmgStart(Timers,TMG_SuComputeStep);
  float dtsq_05=0.5f*dt*dt;
  VerletStep++;
  if(VerletStep<VerletSteps){
    const float twodt=dt+dt;
    cusph::ComputeStepVerlet(rhopbound,WithFloating,Np,Npb,Velg,VelM1g,RhopM1g,Idpg,Arg,Aceg,VelXcorg,dt,twodt,Eps,MovLimit,Posg,Codeg,VelM1g,RhopM1g,Rhop0); 
  }
  else{
    cusph::ComputeStepVerlet(rhopbound,WithFloating,Np,Npb,Velg,Velg,Rhopg,Idpg,Arg,Aceg,VelXcorg,dt,dt,Eps,MovLimit,Posg,Codeg,VelM1g,RhopM1g,Rhop0); 
    VerletStep=0;
  }
  //-New values are computed in VelM1g & RhopM1g.
  swap(Velg,VelM1g);         //Swaps Velg <= VelM1g.
  swap(Rhopg,RhopM1g);       //Swaps Rhopg <= RhopM1g.
  if(CaseNmoving)cudaMemset(Velg,0,sizeof(float3)*Npb); //Velg[]=0 for boundaries.
  TmgStop(Timers,TMG_SuComputeStep);
}

//==============================================================================
/// Updates particles according to forces and dt using SYMPLECTIC (Predictor step).
//==============================================================================
void JSphGpu::ComputeSymplecticPre(bool rhopbound,float dt){
  TmgStart(Timers,TMG_SuComputeStep);
  //-Assigns memory to variables Pre.
  PosPreg=GpuArrays->ReserveFloat3();
  VelPreg=GpuArrays->ReserveFloat3();
  RhopPreg=GpuArrays->ReserveFloat();
  //-Changes data to variables Pre to compute new data.
  swap(PosPreg,Posg);        //Swaps PosPreg[] <= Posg[].
  swap(VelPreg,Velg);        //Swaps VelPreg[] <= Velg[].
  swap(RhopPreg,Rhopg);      //Swaps RhopPreg[]<= Rhopg[].
  //-Computes new data of particles.
  const float dt05=dt*.5f;
  cusph::ComputeStepSymplecticPre(rhopbound,WithFloating,Np,Npb,Idpg,PosPreg,VelPreg,RhopPreg,Arg,VelXcorg,Aceg,dt05,Eps,MovLimit,Posg,Codeg,Velg,Rhopg,Rhop0); 
  cudaMemcpy(Posg,PosPreg,Npb*sizeof(float3),cudaMemcpyDeviceToDevice);
  cudaMemcpy(Velg,VelPreg,Npb*sizeof(float3),cudaMemcpyDeviceToDevice);
  TmgStop(Timers,TMG_SuComputeStep);
}

//==============================================================================
/// Updates particles according to forces and dt using SYMPLECTIC (Corrector step).
//==============================================================================
void JSphGpu::ComputeSymplecticCorr(bool rhopbound,float dt){
  TmgStart(Timers,TMG_SuComputeStep);
  const float dt05=dt*.5f;
  cusph::ComputeStepSymplecticCor(rhopbound,WithFloating,Np,Npb,Idpg,PosPreg,VelPreg,RhopPreg,Arg,Aceg,dt05,dt,MovLimit,Posg,Codeg,Velg,Rhopg,Rhop0); 
  if(CaseNmoving)cudaMemset(Velg,0,sizeof(float3)*Npb); //Velg[]=0 for boundaries.
  //-Releases memroy assigned to variables Pre in ComputeSymplecticPre().
  GpuArrays->Free(PosPreg);   PosPreg=NULL;
  GpuArrays->Free(VelPreg);   VelPreg=NULL;
  GpuArrays->Free(RhopPreg);  RhopPreg=NULL;
  TmgStop(Timers,TMG_SuComputeStep);
}

//==============================================================================
/// Computes a variable DT.
//==============================================================================
float JSphGpu::DtVariable(){
  //float dt=(CsoundMax||ViscDtMax? CFLnumber*(H/(CsoundMax+H*ViscDtMax)): FLT_MAX);
  
  float dt1=(CsoundMax||ViscDtMax? CFLnumber*(H/(CsoundMax+H*ViscDtMax)): FLT_MAX);
  float dt2=(ForceDtMax?           CFLnumber*(sqrt(H/ForceDtMax))		: FLT_MAX);
  float dt3=(ViscoDtMax?           CFLnumber*(H*H*ViscoDtMax )			: FLT_MAX);
  float dt=min(dt1,dt2); 
        dt=min(dt3,dt);

  if(DtFixed)dt=DtFixed->GetDt(TimeStep,dt);
  //char cad[512]; sprintf(cad,"____Dt[%u]:%f  csound:%f  viscdt:%f",Nstep,dt,CsoundMax,ViscDt); Log->Print(cad);
  if(dt<DtMin){ dt=DtMin; DtModif++; }
  return(dt);
}


//==============================================================================
/// Processes movement of moving boundary particles.
//==============================================================================
void JSphGpu::RunMotion(float stepdt){
  TmgStart(Timers,TMG_SuMotion);
  if(Motion->ProcesTime(TimeStep+MotionTimeMod,stepdt)){
    unsigned nmove=Motion->GetMovCount();
    //{ char cad[256]; sprintf(cad,"----RunMotion[%u]>  nmove:%u",Nstep,nmove); Log->Print(cad); }
    if(nmove){
      cudaMemset(RidpMoving,255,CaseNmoving*sizeof(unsigned));  //-Assigns UINT_MAX values.
      cusph::CalcRidp(Npb,0,CaseNfixed,CaseNfixed+CaseNmoving,Idpg,RidpMoving);
      //-Movevement of boundary particles.
      for(unsigned c=0;c<nmove;c++){
        unsigned ref;
        tfloat3 mvsimple;
        tmatrix4f mvmatrix;
        if(Motion->GetMov(c,ref,mvsimple,mvmatrix)){  //-Simple movement.
          mvsimple=OrderCode(mvsimple);
          const unsigned pini=MotionObjBegin[ref]-CaseNfixed,np=MotionObjBegin[ref+1]-MotionObjBegin[ref];
          cusph::MoveLinBound(Simulate2D,np,pini,mvsimple,mvsimple/TFloat3(stepdt),RidpMoving,Posg,Velg,Codeg,MovLimit);
        }
        else{                                         //-Movement with matrix.
          mvmatrix=OrderCode(mvmatrix);
          const unsigned pini=MotionObjBegin[ref]-CaseNfixed,np=MotionObjBegin[ref+1]-MotionObjBegin[ref];
          //sprintf(Cad,"--->pini:%u np:%u",pini,np); Log->Print(Cad);
          cusph::MoveMatBound(Simulate2D,np,pini,mvmatrix,stepdt,RidpMoving,Posg,Velg,Codeg,MovLimit);
        } 
      }
      BoundChanged=true;
    }
  }
  TmgStop(Timers,TMG_SuMotion);
}

//==============================================================================
/// Shows active timers.
//==============================================================================
void JSphGpu::ShowTimers(bool onlyfile){
  JLog2::TpMode_Out mode=(onlyfile? JLog2::Out_File: JLog2::Out_ScrFile);
  Log->Print("\n[GPU Timers]",mode);
  if(!SvTimers)Log->Print("none",mode);
  else for(unsigned c=0;c<TimerGetCount();c++)if(TimerIsActive(c))Log->Print(TimerToText(c),mode);
}

//==============================================================================
/// Returns string with name and values of active timers.
//==============================================================================
void JSphGpu::GetTimersInfo(std::string &hinfo,std::string &dinfo)const{
  for(unsigned c=0;c<TimerGetCount();c++)if(TimerIsActive(c)){
    hinfo=hinfo+";"+TimerGetName(c);
    dinfo=dinfo+";"+fun::FloatStr(TimerGetValue(c)/1000.f);
  }
}

//==============================================================================
// Stores VTK file with particle data.
//==============================================================================
/*void JSphGpu::DgSaveVtkParticlesGpu(std::string filename,int numfile,unsigned pini,unsigned pfin,const float3 *posg,const byte *checkg,const unsigned *idpg,const float3 *velg,const float *rhopg){
  //-Allocates basic memory.
  const unsigned n=pfin-pini;
  tfloat3 *pos=new tfloat3[n];
  byte *check=NULL;
  unsigned *idp=NULL;
  tfloat3 *vel=NULL;
  float *rhop=NULL;
  if(checkg)check=new byte[n];
  if(idpg)idp=new unsigned[n];
  if(velg)vel=new tfloat3[n];
  if(rhopg)rhop=new float[n];
  //-Recovers data from GPU.
  cudaMemcpy(pos,posg+pini,sizeof(float3)*n,cudaMemcpyDeviceToHost);
  if(checkg)cudaMemcpy(check,checkg+pini,sizeof(byte)*n,cudaMemcpyDeviceToHost);
  if(idpg)cudaMemcpy(idp,idpg+pini,sizeof(unsigned)*n,cudaMemcpyDeviceToHost);
  if(velg)cudaMemcpy(vel,velg+pini,sizeof(float3)*n,cudaMemcpyDeviceToHost);
  if(rhopg)cudaMemcpy(rhop,rhopg+pini,sizeof(float)*n,cudaMemcpyDeviceToHost);
  //-Generates VTK file.
  DgSaveVtkParticlesCpu(filename,numfile,0,n,pos,check,idp,vel,rhop);
  //-Releases memory.
  delete[] pos;
  delete[] check;
  delete[] idp;
  delete[] vel;
  delete[] rhop;
}*/

//==============================================================================
/// Stores CSV file with particle data.
//==============================================================================
/*void JSphGpu::DgSaveCsvParticlesGpu(std::string filename,int numfile,unsigned pini,unsigned pfin,std::string head,const float3 *posg,const unsigned *idpg,const float3 *velg,const float *rhopg,const float *arg,const float3 *aceg,const float3 *vcorrg){
  const char met[]="DgSaveCsvParticlesGpu";
  //-Allocates basic memory.
  const unsigned n=pfin-pini;
  unsigned *idp=NULL;  if(idpg)idp=new unsigned[n];
  tfloat3 *pos=NULL;   if(posg)pos=new tfloat3[n];
  tfloat3 *vel=NULL;   if(velg)vel=new tfloat3[n];
  float *rhop=NULL;    if(rhopg)rhop=new float[n];
  float *ar=NULL;      if(arg)ar=new float[n];
  tfloat3 *ace=NULL;   if(aceg)ace=new tfloat3[n];
  tfloat3 *vcorr=NULL; if(vcorrg)vcorr=new tfloat3[n];
  //-Recovers data from GPU.
  if(idpg)cudaMemcpy(idp,idpg+pini,sizeof(unsigned)*n,cudaMemcpyDeviceToHost);
  if(posg)cudaMemcpy(pos,posg+pini,sizeof(float3)*n,cudaMemcpyDeviceToHost);
  if(velg)cudaMemcpy(vel,velg+pini,sizeof(float3)*n,cudaMemcpyDeviceToHost);
  if(rhopg)cudaMemcpy(rhop,rhopg+pini,sizeof(float)*n,cudaMemcpyDeviceToHost);
  if(arg)cudaMemcpy(ar,arg+pini,sizeof(float)*n,cudaMemcpyDeviceToHost);
  if(aceg)cudaMemcpy(ace,aceg+pini,sizeof(float3)*n,cudaMemcpyDeviceToHost);
  if(vcorrg)cudaMemcpy(vcorr,vcorrg+pini,sizeof(float3)*n,cudaMemcpyDeviceToHost);
  CheckCudaError(met,"Failed copying data from GPU.");
  //-Generates CSV file.
  DgSaveCsvParticlesCpu(filename,numfile,0,n,head,pos,idp,vel,rhop,ar,ace,vcorr);
  //-Releases memory.
  delete[] idp;
  delete[] pos;
  delete[] vel;
  delete[] rhop;
  delete[] ar;
  delete[] ace;
  delete[] vcorr;
}*/




