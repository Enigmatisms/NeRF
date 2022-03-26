/**
 * @file error_check.cuh
 * @author Unknown. Adopted from stackoverflow
 * @brief Output error message in CUDA programs
 */
#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
__host__ static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

__host__ static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cout << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line <<
			std::endl;
	exit (1);
}