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
    // TODO: HW 4.1. Your code below.
    v = v + dt * (0, -9.8, 0)  // gravity changes velocity
                x = x + dt * v // move with the NEW velocity (symplectic Euler)
                            if (x.y < -0.97) {                // went through the floor?
        x.y = -0.97; //   put it back on the floor
        v.y = -v.y;  //   bounce: flip the vertical velocity
    }
}
