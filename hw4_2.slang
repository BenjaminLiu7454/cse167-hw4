[shader("compute")]
[numthreads(64, 1, 1)]
void predict(uint tid : SV_DispatchThreadID,
             // Updated position
             RWStructuredBuffer<float3> positions,
             // Store the position before update
             RWStructuredBuffer<float3> old_positions,
             RWStructuredBuffer<float3> velocities,
             StructuredBuffer<float> inv_masses,
             uniform uint num_particles,
             // time step size
             uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    float3 p = positions[tid];
    old_positions[tid] = p;

    if (inv_masses[tid] == 0.0) { // pinned
        return;
    }

    float3 v = velocities[tid];

    v += dt * float3(0.0, -9.8, 0.0); // gravity
    p += dt * v;                      // move with the new velocity

    positions[tid] = p; // write back
    velocities[tid] = v;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<float> rest_lengths,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }
    int id = tid;
    float3 p = positions_in[id];
    float wi = inv_masses[id];

    float3 correction = float3(0.0, 0.0, 0.0);
    int count = 0; // how many corrections to average out
    for (int k = 0; k < 2; ++k) {
        int j = (k == 0) ? id - 1 : id + 1; // left then right neighbor
        if (j < 0 || j >= int(num_particles)) continue; //end of rope

        float l = rest_lengths[min(id, j)];       // C(wi, wj) = |wj − wi| − l = 0
        float3 pj = positions_in[j];
        float wj = inv_masses[j];
        float3 d = pj - p;
        float len = length(d);
        if (wi + wj == 0.0 || len == 0.0) continue; // avoid div by 0
        correction += wi / (wi + wj) * (len - l) * (d / len);
        count++;
    }
    if (count > 0) {
        correction /= float(count); // average the corrections
    }
    positions_out[id] = p + correction;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     // time step size
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    float3 o = old_positions[tid];
    float3 p = positions[tid];
    float3 v = velocities[tid];
    p.y = max(p.y, -0.97);
    v = (p - o) / dt;

    positions[tid] = p; // write back
    velocities[tid] = v;
}
