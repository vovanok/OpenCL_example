// Vector additional kernel
__kernel void VectorAdd(__global const float* a, __global const float* b, __global float* c, int numElements)
{
    // Get global index for current thread
    int glIndex = get_global_id(0);

    // Out of vector bound check
    if (glIndex >= numElements)
        return;
    
    // Vector's elements addition
    c[glIndex] = a[glIndex] + b[glIndex];
}
