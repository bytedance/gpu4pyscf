
#pragma once

#define IS_FLOAT(type) (sizeof(type) == sizeof(float))
#define IS_DOUBLE(type) (sizeof(type) == sizeof(double))

template<typename FloatType>
class MixedPrecisionOperator {};

template<>
class MixedPrecisionOperator<double> {
    public:
    static __device__ double fp_exp(const double x) { return exp(x); }
    static __device__ double fp_erf(const double x) { return erf(x); }
    static __device__ double fp_sqrt(const double x) { return sqrt(x); }
};

template<>
class MixedPrecisionOperator<float> {
    public:
    static __device__ float fp_exp(const float x) { return expf(x); }
    static __device__ float fp_erf(const float x) { return erff(x); }
    static __device__ float fp_sqrt(const float x) { return sqrtf(x); }
};