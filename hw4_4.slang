[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         // The CSR matrix data structure
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         // Rest lengthes of the neighbor edges
                         StructuredBuffer<float> neighbor_rest,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }
    int i = int(tid);
    float3 xi = positions_in[i];
    float wi = inv_masses[i];

    float3 correction = float3(0.0, 0.0, 0.0);
    int count = 0;

    for (uint k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
        int j = int(neighbors[k]);
        float l = neighbor_rest[k]; // rest length for this spring

        float3 d = positions_in[j] - xi;
        float wj = inv_masses[j];
        float len = length(d);
        if (wi + wj == 0.0 || len == 0.0) continue;

        correction += wi / (wi + wj) * (len - l) * (d / len);
        count++;
    }

    if (count > 0) correction /= float(count);
    positions_out[i] = xi + correction;
}
