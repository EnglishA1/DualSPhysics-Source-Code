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

/// \file Types.h \brief Defines specific types for the SPH application.

#ifndef _Types_
#define _Types_

#include "TypesDef.h"
#include <algorithm>

//#define _CODE_FAST         //-Compilation without timers.

//#define DT_ALLPARTICLES    //-Activates/deactivates the use of all particles (not only the fluid ones) to compute the value of dt. 

//#define _WITHOMP             ///<Enables/Disables OpenMP.   
#define MAXTHREADS_OMP 64
#define STRIDE_OMP 200

#define USE_SYMMETRY true    //-Activates/deactivates symmetry in force computation.

//#define _WITHGPU 1 

#define BORDER_MAP 0.001f

#define OVERPI 0.318309886   ///<Value of 1/PI.

//#define RHOPCERO 1000.f
//#define OVERRHOPCERO 0.001f

#define MK_RANGE 256         ///<Maximum amount of MK label of the particles.

//-Coce of the particles:
#define CODE_MASKOUT 0xe000       //-Bits for out:     1110 0000 0000 0000
#define CODE_OUT_OK 0x0
#define CODE_OUT_MOVE 0x2000
#define CODE_OUT_POS 0x4000
#define CODE_OUT_RHOP 0x6000
#define CODE_MASKTYPEVALUE 0x1fff //-Bits for type:    0001 1111 1111 1111
#define CODE_MASKTYPE 0x1800      //-Bits for type:    0001 1000 0000 0000
#define CODE_TYPE_FIXED 0x0       //---Particles fixed:  0-2047
#define CODE_TYPE_MOVING 0x800    //---Particles moving: 2048-4095
#define CODE_TYPE_FLOATING 0x1000 //---Particles moving: 4096-6143
#define CODE_TYPE_FLUID 0x1800    //---Particles fluid:  6144-8191
#define CODE_MASKVALUE 0x7ff      //-Bits type-value: 0000 0111 1111 1111  Range:0-2047

#define CODE_SetOutOk(code)    (code&(~CODE_MASKOUT))
#define CODE_SetOutPos(code)   (CODE_SetOutOk(code)|CODE_OUT_POS)
#define CODE_SetOutMove(code)  (CODE_SetOutOk(code)|CODE_OUT_MOVE)
#define CODE_SetOutRhop(code)  (CODE_SetOutOk(code)|CODE_OUT_RHOP)
#define CODE_GetOutValue(code) (code&CODE_MASKOUT)

#define CODE_GetType(code) (code&CODE_MASKTYPE)
#define CODE_GetTypeValue(code) (code&CODE_MASKVALUE)
#define CODE_GetValue(code) (code&CODE_MASKTYPEVALUE)


/// Structure with the information of the floating object.
typedef struct{
  unsigned begin;            ///<First particle of the floating object.
  unsigned count;            ///<Number of objects.
  float mass;                ///<Mass of the object.
  float massp;               ///<Mass of the particle of the floating object.
  tfloat3 inertia;           ///<Inertial momentum of the object.
  tfloat3 center;            ///<Center of the object.
  tfloat3 fvel;              ///<Linear velocity of the object.
  tfloat3 fomega;            ///<Angular velocity of the object
}StFloatingData;

///Controls the output of information on the screen and/or log.
typedef enum{ 
    MOUT_ScrFile=3,          ///<Output on the screen and log.
    MOUT_File=2,             ///<Output in the log.
    MOUT_Screen=1,           ///<Output on the screen.
    MOUT_None=0              ///<No output.
}TpModeOut;   

///Options of the format of output files.
typedef enum{ 
  SDAT_Binx=1,               ///<BYNARY format .bi2
  SDAT_Vtk=2,                ///<VTK format .vtk
  SDAT_Csv=4,                ///<CSV format .csv
  SDAT_Info=8,           ///<Info v4 format (SPHysics code).
  SDAT_None=0            
}TpSaveDat;                 

///Types of step algorithm.
typedef enum{ 
  STEP_Symplectic=2,         ///<Symplectic algorithm.
  STEP_Verlet=1,             ///<Verlet algorithm.
  STEP_None=0 
}TpStep;                    

///Types of kernel function.
typedef enum{ 
  KERNEL_Wendland=2,         ///<Wendland kernel.
  KERNEL_Cubic=1,            ///<Cubic Spline kernel.
  KERNEL_None=0 
}TpKernel;                  

///Types of viscosity treatment.
typedef enum{
  VISCO_SumSPS=3,
  VISCO_LaminarSPS=2,        ///<Laminar viscosity and Sub-Partice Scale Turbulence.
  VISCO_Artificial=1,        ///<Artificial viscosity.
  VISCO_None=0 
}TpVisco;            

///Types of interaction.
typedef enum{ 
  INTER_Shepard=3,           ///<Interaction to compute the new density values when using Shepard density filter.
  INTER_ForcesCorr=2,        ///<Interaction to compute forces using the corrector step of Symplectic algorithm (where XSPH variant is not applied).
  INTER_Forces=1             ///<Interaction to compute forces using the Verlet algorithm and the predictor step of Symplectic algorithm . 
}TpInter;   

///Types of Delta-SPH approach.
typedef enum{ 
  DELTA_DBCExt=3,            ///<DeltaSPH approach applied in case of Periodic Boundary Conditions or new multiGPU implementation.
  DELTA_DBC=2,               ///<DeltaSPH approach applied only for fluid particles that are not interaction with boundaries (DBC).
  DELTA_None=0
}TpDeltaSph;

///Types of interaction with periodic zones.
typedef enum{ 
  IPERI_Z=3,
  IPERI_Y=2,
  IPERI_X=1 
}TpInterPeri; 

///Types of particles.
typedef enum{ 
    PART_BoundFx=1,          ///<Fixed boundary particles.
    PART_BoundMv=2,          ///<Moving boundary particles.
    PART_BoundFx_BoundMv=3,  ///<Both fixed and moving boundary particles.
    PART_BoundFt=4,          ///<Floating boundary particles.
    PART_Fluid=8,            ///<Fluid particles.
    PART_BoundFt_Fluid=12    ///<Both floating and fluid particles.
}TpParticle;

///Types of execution with or without OpenMP.
typedef enum{ 
    OMPM_Single,             ///<Single execution with one core of CPU without OpenMP.
    OMPM_Dynamic,            ///<Multiple-core execution using OpenMP with dynamic load balancing.
    OMPM_Static              ///<Multiple-core execution using OpenMP with staticc load balancing.
}TpOmpMode;  

///Order of the axis to reorder particles in cells.
typedef enum{ 
    ORDER_None=0,
    ORDER_XYZ=1,
    ORDER_XZY=2,
    ORDER_YXZ=3,
    ORDER_YZX=4,
    ORDER_ZXY=5,
    ORDER_ZYX=6 
}TpCellOrder;  

///Returns the name of the type of interaction with or without OpenMP in text format.
inline const char* GetNameOmpMode(TpOmpMode mode){
  switch(mode){
    case OMPM_Single:   return("Single");
    case OMPM_Dynamic:  return("Dynamic");
    case OMPM_Static:   return("Static");
  }
  return("???");
}

///Returns the name of the CellOrder in text format.
inline const char* GetNameCellOrder(TpCellOrder cellorder){
  switch(cellorder){
    case ORDER_XYZ:   return("XYZ");
    case ORDER_XZY:   return("XZY");
    case ORDER_YXZ:   return("YXZ");
    case ORDER_YZX:   return("YZX");
    case ORDER_ZXY:   return("ZXY");
    case ORDER_ZYX:   return("ZYX");
  }
  return("???");
}

inline tuint3 GetCodeCellOrder(TpCellOrder cellorder){
  switch(cellorder){
    case ORDER_XYZ:   return(TUint3(1,2,3));
    case ORDER_XZY:   return(TUint3(1,3,2));
    case ORDER_YXZ:   return(TUint3(2,1,3));
    case ORDER_YZX:   return(TUint3(2,3,1));
    case ORDER_ZXY:   return(TUint3(3,1,2));
    case ORDER_ZYX:   return(TUint3(3,2,1));
  }
  return(TUint3(1,2,3));
}


inline TpCellOrder GetDecodeOrder(TpCellOrder order){
  switch(order){
    case ORDER_XYZ:   return(ORDER_XYZ);
    case ORDER_XZY:   return(ORDER_XZY);
    case ORDER_YXZ:   return(ORDER_YXZ);
    case ORDER_YZX:   return(ORDER_ZXY);
    case ORDER_ZXY:   return(ORDER_YZX);
    case ORDER_ZYX:   return(ORDER_ZYX);
  }
  return(ORDER_None);
}



/////Returns the reordered tfloat3 value.
//inline tfloat3 ReOrderXZY(const tfloat3 &v){ return(TFloat3(v.x,v.z,v.y)); }
//inline tfloat3 ReOrderYXZ(const tfloat3 &v){ return(TFloat3(v.y,v.x,v.z)); }
//inline tfloat3 ReOrderYZX(const tfloat3 &v){ return(TFloat3(v.y,v.z,v.x)); }
//inline tfloat3 ReOrderZXY(const tfloat3 &v){ return(TFloat3(v.z,v.x,v.y)); }
//inline tfloat3 ReOrderZYX(const tfloat3 &v){ return(TFloat3(v.z,v.y,v.x)); }

///Returns the reordered tuint3 value.
inline tuint3 ReOrderXZY(const tuint3 &v){ return(TUint3(v.x,v.z,v.y)); }
inline tuint3 ReOrderYXZ(const tuint3 &v){ return(TUint3(v.y,v.x,v.z)); }
inline tuint3 ReOrderYZX(const tuint3 &v){ return(TUint3(v.y,v.z,v.x)); }
inline tuint3 ReOrderZXY(const tuint3 &v){ return(TUint3(v.z,v.x,v.y)); }
inline tuint3 ReOrderZYX(const tuint3 &v){ return(TUint3(v.z,v.y,v.x)); }

///Returns the reordered tmatrix4f matrix.
inline void ReOrderXZY(tmatrix4f &x){ std::swap(x.a12,x.a13); std::swap(x.a21,x.a31); std::swap(x.a22,x.a33); std::swap(x.a23,x.a32); std::swap(x.a24,x.a34); }
inline void ReOrderYXZ(tmatrix4f &x){ std::swap(x.a11,x.a21); std::swap(x.a12,x.a22); std::swap(x.a13,x.a23); std::swap(x.a14,x.a24); std::swap(x.a11,x.a12); std::swap(x.a21,x.a22); std::swap(x.a31,x.a32); }
inline void ReOrderYZX(tmatrix4f &x){ ReOrderYXZ(x); ReOrderXZY(x); }
inline void ReOrderZXY(tmatrix4f &x){ ReOrderXZY(x); ReOrderYXZ(x); }
inline void ReOrderZYX(tmatrix4f &x){ std::swap(x.a11,x.a31); std::swap(x.a12,x.a32); std::swap(x.a13,x.a33); std::swap(x.a14,x.a34); std::swap(x.a11,x.a13); std::swap(x.a21,x.a23); std::swap(x.a31,x.a33); }


///Devuelve valor tfloat3 reordenado.
///Returns reordered tfloat3 value.
inline tfloat3 ReOrderXZY(const tfloat3 &v){ return(TFloat3(v.x,v.z,v.y)); }
inline tfloat3 ReOrderYXZ(const tfloat3 &v){ return(TFloat3(v.y,v.x,v.z)); }
inline tfloat3 ReOrderYZX(const tfloat3 &v){ return(TFloat3(v.y,v.z,v.x)); }
inline tfloat3 ReOrderZXY(const tfloat3 &v){ return(TFloat3(v.z,v.x,v.y)); }
inline tfloat3 ReOrderZYX(const tfloat3 &v){ return(TFloat3(v.z,v.y,v.x)); }

///Devuelve valor tdouble3 reordenado.
///Returns reordered tdouble3 value.
inline tdouble3 ReOrderXZY(const tdouble3 &v){ return(TDouble3(v.x,v.z,v.y)); }
inline tdouble3 ReOrderYXZ(const tdouble3 &v){ return(TDouble3(v.y,v.x,v.z)); }
inline tdouble3 ReOrderYZX(const tdouble3 &v){ return(TDouble3(v.y,v.z,v.x)); }
inline tdouble3 ReOrderZXY(const tdouble3 &v){ return(TDouble3(v.z,v.x,v.y)); }
inline tdouble3 ReOrderZYX(const tdouble3 &v){ return(TDouble3(v.z,v.y,v.x)); }

/////Devuelve valor tuint3 reordenado.
/////Returns reordered tuint3 value.
//inline tuint3 ReOrderXZY(const tuint3 &v){ return(TUint3(v.x,v.z,v.y)); }
//inline tuint3 ReOrderYXZ(const tuint3 &v){ return(TUint3(v.y,v.x,v.z)); }
//inline tuint3 ReOrderYZX(const tuint3 &v){ return(TUint3(v.y,v.z,v.x)); }
//inline tuint3 ReOrderZXY(const tuint3 &v){ return(TUint3(v.z,v.x,v.y)); }
//inline tuint3 ReOrderZYX(const tuint3 &v){ return(TUint3(v.z,v.y,v.x)); }

///Devuelve valor tfloat4 reordenado.
///Returns reordered tfloat4 value.
inline tfloat4 ReOrderXZY(const tfloat4 &v){ return(TFloat4(v.x,v.z,v.y,v.w)); }
inline tfloat4 ReOrderYXZ(const tfloat4 &v){ return(TFloat4(v.y,v.x,v.z,v.w)); }
inline tfloat4 ReOrderYZX(const tfloat4 &v){ return(TFloat4(v.y,v.z,v.x,v.w)); }
inline tfloat4 ReOrderZXY(const tfloat4 &v){ return(TFloat4(v.z,v.x,v.y,v.w)); }
inline tfloat4 ReOrderZYX(const tfloat4 &v){ return(TFloat4(v.z,v.y,v.x,v.w)); }

///Reordena matriz tmatrix4f.
///Reorders tmatrix4f matrix.
inline void ReOrderXZY(tmatrix4d &x){ std::swap(x.a12,x.a13); std::swap(x.a21,x.a31); std::swap(x.a22,x.a33); std::swap(x.a23,x.a32); std::swap(x.a24,x.a34); }
inline void ReOrderYXZ(tmatrix4d &x){ std::swap(x.a11,x.a21); std::swap(x.a12,x.a22); std::swap(x.a13,x.a23); std::swap(x.a14,x.a24); std::swap(x.a11,x.a12); std::swap(x.a21,x.a22); std::swap(x.a31,x.a32); }
inline void ReOrderYZX(tmatrix4d &x){ ReOrderYXZ(x); ReOrderXZY(x); }
inline void ReOrderZXY(tmatrix4d &x){ ReOrderXZY(x); ReOrderYXZ(x); }
inline void ReOrderZYX(tmatrix4d &x){ std::swap(x.a11,x.a31); std::swap(x.a12,x.a32); std::swap(x.a13,x.a33); std::swap(x.a14,x.a34); std::swap(x.a11,x.a13); std::swap(x.a21,x.a23); std::swap(x.a31,x.a33); }


///Devuelve valor tfloat3 reordenado.
///Returns reordered tfloat3 value.
inline tfloat3 OrderCodeValue(TpCellOrder order,const tfloat3 &v){
  switch(order){
    case ORDER_XZY:   return(ReOrderXZY(v));
    case ORDER_YXZ:   return(ReOrderYXZ(v));
    case ORDER_YZX:   return(ReOrderYZX(v));
    case ORDER_ZXY:   return(ReOrderZXY(v));
    case ORDER_ZYX:   return(ReOrderZYX(v));
  }
  return(v);
}

///Devuelve valor tfloat3 en el orden original.
///Returns tfloat3 value in the original order.
inline tfloat3 OrderDecodeValue(TpCellOrder order,const tfloat3 &v){ return(OrderCodeValue(GetDecodeOrder(order),v)); }

///Returns the reordered tfloat3 value according to a given order.
inline tdouble3 OrderCodeValue(TpCellOrder order,const tdouble3 &v){
  switch(order){
    case ORDER_XZY:   return(ReOrderXZY(v));
    case ORDER_YXZ:   return(ReOrderYXZ(v));
    case ORDER_YZX:   return(ReOrderYZX(v));
    case ORDER_ZXY:   return(ReOrderZXY(v));
    case ORDER_ZYX:   return(ReOrderZYX(v));
  }
  return(v);
}

///Retunrs the original order of tfloat3 value according to a given order.
inline tdouble3 OrderDecodeValue(TpCellOrder order,const tdouble3 &v){ return(OrderCodeValue(GetDecodeOrder(order),v)); }

///Returns the reordered tuint3 value according to a given order.
inline tuint3 OrderCodeValue(TpCellOrder order,const tuint3 &v){
  switch(order){
    case ORDER_XZY:   return(ReOrderXZY(v));
    case ORDER_YXZ:   return(ReOrderYXZ(v));
    case ORDER_YZX:   return(ReOrderYZX(v));
    case ORDER_ZXY:   return(ReOrderZXY(v));
    case ORDER_ZYX:   return(ReOrderZYX(v));
  }
  return(v);
}

///Retunrs the original order of tuint3 value according to a given order.
inline tuint3 OrderDecodeValue(TpCellOrder order,const tuint3 &v){ return(OrderCodeValue(GetDecodeOrder(order),v)); }

///Returns the reordered tmatrix4d matrix according to a given order.
inline tmatrix4d OrderCodeValue(TpCellOrder order,tmatrix4d x){
  switch(order){
    case ORDER_XZY:   ReOrderXZY(x);   break;
    case ORDER_YXZ:   ReOrderYXZ(x);   break;
    case ORDER_YZX:   ReOrderYZX(x);   break;
    case ORDER_ZXY:   ReOrderZXY(x);   break;
    case ORDER_ZYX:   ReOrderZYX(x);   break;
  }
  return(x);
} 

///Returns the reordered tmatrix4f matrix according to a given order.
inline tmatrix4f OrderCodeValue(TpCellOrder order,tmatrix4f x){
  switch(order){
    case ORDER_XZY:   ReOrderXZY(x);   break;
    case ORDER_YXZ:   ReOrderYXZ(x);   break;
    case ORDER_YZX:   ReOrderYZX(x);   break;
    case ORDER_ZXY:   ReOrderZXY(x);   break;
    case ORDER_ZYX:   ReOrderZYX(x);   break;
  }
  return(x);
} 

/*
///Returns the reordered tfloat3 value according to a given order.
inline tfloat3 OrderCodeValue(TpCellOrder order,const tfloat3 &v){
  switch(order){
    case ORDER_XZY:   return(ReOrderXZY(v));
    case ORDER_YXZ:   return(ReOrderYXZ(v));
    case ORDER_YZX:   return(ReOrderYZX(v));
    case ORDER_ZXY:   return(ReOrderZXY(v));
    case ORDER_ZYX:   return(ReOrderZYX(v));
  }
  return(v);
} 
///Retunrs the original order of tfloat3 value according to a given order.
inline tfloat3 OrderDecodeValue(TpCellOrder order,const tfloat3 &v){ return(OrderCodeValue(GetDecodeOrder(order),v)); }

///Returns the reordered tuint3 value according to a given order.
inline tuint3 OrderCodeValue(TpCellOrder order,const tuint3 &v){
  switch(order){
    case ORDER_XZY:   return(ReOrderXZY(v));
    case ORDER_YXZ:   return(ReOrderYXZ(v));
    case ORDER_YZX:   return(ReOrderYZX(v));
    case ORDER_ZXY:   return(ReOrderZXY(v));
    case ORDER_ZYX:   return(ReOrderZYX(v));
  }
  return(v);
} 
///Retunrs the original order of tuint3 value according to a given order.
inline tuint3 OrderDecodeValue(TpCellOrder order,const tuint3 &v){ return(OrderCodeValue(GetDecodeOrder(order),v)); }

///Returns the reordered tmatrix4f matrix according to a given order.
inline tmatrix4f OrderCodeValue(TpCellOrder order,tmatrix4f x){
  switch(order){
    case ORDER_XZY:   ReOrderXZY(x);   break;
    case ORDER_YXZ:   ReOrderYXZ(x);   break;
    case ORDER_YZX:   ReOrderYZX(x);   break;
    case ORDER_ZXY:   ReOrderZXY(x);   break;
    case ORDER_ZYX:   ReOrderZYX(x);   break;
  }
  return(x);
} 
*/

///Modes of cells division.
typedef enum{ 
   CELLMODE_None=0
  ,CELLMODE_2H=1             ///<Cells of size 2h.
  ,CELLMODE_H=2              ///<Cells of size h.
}TpCellMode; 

///Returns the name of the CELLMODE in text format.
inline const char* GetNameCellMode(TpCellMode cellmode){
  switch(cellmode){
    case CELLMODE_2H:      return("2H");
    case CELLMODE_H:       return("H");
  }
  return("???");
}


#endif


