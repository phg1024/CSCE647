#pragma once

struct AbsorptionAndScatteringProp {
	__device__ AbsorptionAndScatteringProp():absortionCoeffs(make_float3(0.0)), reducedScatteringCoeffs(0.0){}
	__device__ void reset() { absortionCoeffs = make_float3(0.0); reducedScatteringCoeffs = 0.0; }
	float3 absortionCoeffs;
	float  reducedScatteringCoeffs;
};

struct Fresnel {
	float reflectionCoeffs;
	float tranmissionCoeffs;
};