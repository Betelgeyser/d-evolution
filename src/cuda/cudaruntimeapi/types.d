/**
 * Copyright © 2018 Sergei Iurevich Filippov, All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Cuda runtime API types.
 */
module cuda.cudaruntimeapi.types;

immutable uint cudaMemAttachGlobal = 0x01;
immutable uint cudaMemAttachHost   = 0x02;

alias cudaStream_t = CUstream_st*;
private struct CUstream_st;

struct dim3
{
    uint x = 1;
    uint y = 1;
    uint z = 1;
    
    this(uint x, uint y, uint z)
    {
    	this.x = x;
    	this.y = y;
    	this.z = z;
    }
}

enum cudaError_t
{
	cudaSuccess                          =     0,
	cudaErrorMissingConfiguration        =     1,
	cudaErrorMemoryAllocation            =     2,
	cudaErrorInitializationError         =     3,
	cudaErrorLaunchFailure               =     4,
	cudaErrorPriorLaunchFailure          =     5,
	cudaErrorLaunchTimeout               =     6,
	cudaErrorLaunchOutOfResources        =     7,
	cudaErrorInvalidDeviceFunction       =     8,
	cudaErrorInvalidConfiguration        =     9,
	cudaErrorInvalidDevice               =    10,
	cudaErrorInvalidValue                =    11,
	cudaErrorInvalidPitchValue           =    12,
	cudaErrorInvalidSymbol               =    13,
	cudaErrorMapBufferObjectFailed       =    14,
	cudaErrorUnmapBufferObjectFailed     =    15,
	cudaErrorInvalidHostPointer          =    16,
	cudaErrorInvalidDevicePointer        =    17,
	cudaErrorInvalidTexture              =    18,
	cudaErrorInvalidTextureBinding       =    19,
	cudaErrorInvalidChannelDescriptor    =    20,
	cudaErrorInvalidMemcpyDirection      =    21,
	cudaErrorAddressOfConstant           =    22,
	cudaErrorTextureFetchFailed          =    23,
	cudaErrorTextureNotBound             =    24,
	cudaErrorSynchronizationError        =    25,
	cudaErrorInvalidFilterSetting        =    26,
	cudaErrorInvalidNormSetting          =    27,
	cudaErrorMixedDeviceExecution        =    28,
	cudaErrorCudartUnloading             =    29,
	cudaErrorNotYetImplemented           =    31,
	cudaErrorMemoryValueTooLarge         =    32,
	cudaErrorInvalidResourceHandle       =    33,
	cudaErrorNotReady                    =    34,
	cudaErrorInsufficientDriver          =    35,
	cudaErrorSetOnActiveProcess          =    36,
	cudaErrorInvalidSurface              =    37,
	cudaErrorNoDevice                    =    38,
	cudaErrorECCUncorrectable            =    39,
	cudaErrorSharedObjectSymbolNotFound  =    40,
	cudaErrorSharedObjectInitFailed      =    41,
	cudaErrorUnsupportedLimit            =    42,
	cudaErrorDuplicateVariableName       =    43,
	cudaErrorDuplicateTextureName        =    44,
	cudaErrorDuplicateSurfaceName        =    45,
	cudaErrorDevicesUnavailable          =    46,
	cudaErrorInvalidKernelImage          =    47,
	cudaErrorNoKernelImageForDevice      =    48,
	cudaErrorIncompatibleDriverContext   =    49,
	cudaErrorPeerAccessAlreadyEnabled    =    50,
	cudaErrorPeerAccessNotEnabled        =    51,
	cudaErrorDeviceAlreadyInUse          =    54,
	cudaErrorProfilerDisabled            =    55,
	cudaErrorProfilerNotInitialized      =    56,
	cudaErrorProfilerAlreadyStarted      =    57,
	cudaErrorProfilerAlreadyStopped      =    58,
	cudaErrorAssert                      =    59,
	cudaErrorTooManyPeers                =    60,
	cudaErrorHostMemoryAlreadyRegistered =    61,
	cudaErrorHostMemoryNotRegistered     =    62,
	cudaErrorOperatingSystem             =    63,
	cudaErrorPeerAccessUnsupported       =    64,
	cudaErrorLaunchMaxDepthExceeded      =    65,
	cudaErrorLaunchFileScopedTex         =    66,
	cudaErrorLaunchFileScopedSurf        =    67,
	cudaErrorSyncDepthExceeded           =    68,
	cudaErrorLaunchPendingCountExceeded  =    69,
	cudaErrorNotPermitted                =    70,
	cudaErrorNotSupported                =    71,
	cudaErrorHardwareStackError          =    72,
	cudaErrorIllegalInstruction          =    73,
	cudaErrorMisalignedAddress           =    74,
	cudaErrorInvalidAddressSpace         =    75,
	cudaErrorInvalidPc                   =    76,
	cudaErrorIllegalAddress              =    77,
	cudaErrorInvalidPtx                  =    78,
	cudaErrorInvalidGraphicsContext      =    79,
	cudaErrorNvlinkUncorrectable         =    80,
	cudaErrorJitCompilerNotFound         =    81,
	cudaErrorCooperativeLaunchTooLarge   =    82,
	cudaErrorStartupFailure              =  0x7f,
	cudaErrorApiFailureBase              = 10000
}

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4
}


