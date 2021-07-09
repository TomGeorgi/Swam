#pragma once
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "macros.h"

#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH  64

int add(int a, int b);

void multiply();

void new_multiply();
