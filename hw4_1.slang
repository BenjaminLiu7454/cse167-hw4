[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint tid : SV_DispatchThreadID,
                  RWStructuredBuffer<float3> positions,
                  RWStructuredBuffer<float3> velocities,
                  uniform uint num_particles,
                  // time step size
                  uniform float dt) {
    if (tid >= num_particles) {
        return;
    }

    float3 p = positions[tid]; 
    float3 v = velocities[tid];

    v += dt * float3(0.0, -9.8, 0.0); // gravity (updates velocity)
    p += dt * v;                      // move with the NEW velocity

    if (p.y < -0.97) { // floor bounce (if y is under the floor 0.97)
        p.y = -0.97;   // set position to floor
        v.y = -v.y;    // set y-velocity move upward 
    }

    positions[tid] = p; // write back
    velocities[tid] = v;
}
