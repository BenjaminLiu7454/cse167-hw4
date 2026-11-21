[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         uniform int2 grid_size,
                         uniform float2 spacing) {
    if (tid >= grid_size.x * grid_size.y) {
        return;
    }
    // same logic to  4_2, but using a grid and different constraints
    int id = tid;
    
    int gx = int(id) % grid_size.x;
    int gy = int(id) / grid_size.x;

    float3 p = positions_in[id];
    float wi = inv_masses[id];
    float3 correction = float3(0.0, 0.0, 0.0);
    int count = 0; // how many corrections to average out

    static const int2 offsets[12] = {
        int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1),   // structural
        int2(1, 1), int2(-1, -1), int2(1, -1), int2(-1, 1), // shear
        int2(2, 0), int2(-2, 0), int2(0, 2), int2(0, -2)    // bending
    };

    for (int k = 0; k < 12; k++) {
        int nx = gx + offsets[k].x;
        int ny = gy + offsets[k].y;
        if (nx < 0 || nx >= grid_size.x || ny < 0 || ny >= grid_size.y) continue; // off the cloth
        int j = ny * grid_size.x + nx;
        float l = length(float2(offsets[k].x * spacing.x, offsets[k].y * spacing.y));
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
