
[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      // Plane stores (a, b, c, d) where ax+by+cz=d
                      // a, b, c are normalized to a unit 3D vector
                      StructuredBuffer<float4> planes,
                      // Sphere stores center (xyz) + radius (w)
                      StructuredBuffer<float4> spheres,
                      // Contact plane: dot(n, x) = d, with unit normal n
                      // Set to (0, 0, 0, 0) if there is no contact
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      // How much should we shift the generated
                      // contact plane in the normal direction
                      // (similar to how we shifted the floor by 0.03
                      //  earlier)
                      uniform float collision_distance,
                      // When the signed distance function
                      // evaluates to a value below
                      // contact_margin + collision_distance
                      // (even when it's positive), we count the
                      // particle to be inside the collider and
                      // a collision event has happened.
                      uniform float contact_margin) {

    if (tid >= num_particles) {
        return;
    }

    // Initialize contact plane to no collision.
    contact_planes[tid] = float4(0.0, 0.0, 0.0, 0.0);

    // Static particles do not need collision correction.
    if (inv_masses[tid] == 0.0) {
        return;
    }

    float3 x = positions[tid];

    float min_distance = 1e30;
    float4 best_contact = float4(0.0, 0.0, 0.0, 0.0);

    // Check all planes.
    for (uint i = 0; i < num_planes; i++) {

        float4 plane = planes[i];

        float3 n = plane.xyz;
        float d = plane.w;

        float signed_distance = dot(n, x) - d;

        if (signed_distance < min_distance) {

            min_distance = signed_distance;

            best_contact = float4(
                n,
                d + collision_distance
            );
        }
    }

    // Check all spheres.
    for (uint i = 0; i < num_spheres; i++) {

        float4 sphere = spheres[i];

        float3 center = sphere.xyz;
        float radius = sphere.w;

        float3 direction = x - center;
        float distance_from_center = length(direction);

        float signed_distance = distance_from_center - radius;

        if (signed_distance < min_distance) {

            min_distance = signed_distance;

            // Calculate sphere surface normal.
            float3 n;

            if (distance_from_center > 0.000001) {
                n = direction / distance_from_center;
            }
            else {
                n = float3(0.0, 1.0, 0.0);
            }

            // Calculate tangent plane at sphere surface.
            float3 surface_point = center + radius * n;
            float d = dot(n, surface_point);

            best_contact = float4(
                n,
                d + collision_distance
            );
        }
    }

    // Record collision if within contact margin.
    if (min_distance < contact_margin + collision_distance) {
        contact_planes[tid] = best_contact;
    }
}


[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         StructuredBuffer<float3> old_positions,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         StructuredBuffer<float> neighbor_rest,
                         StructuredBuffer<float4> contact_planes,
                         uniform uint num_particles,
                         uniform float static_friction,
                         uniform float kinetic_friction) {

    if (tid >= num_particles) {
        return;
    }

    int i = int(tid);

    float3 xi = positions_in[i];
    float wi = inv_masses[i];

    // Static particles should not move.
    if (wi == 0.0) {
        positions_out[i] = xi;
        return;
    }


    float3 correction = float3(0.0, 0.0, 0.0);
    int count = 0;

    for (uint k = row_ptr[i]; k < row_ptr[i + 1]; k++) {

        int j = int(neighbors[k]);
        float l = neighbor_rest[k];

        float3 d = positions_in[j] - xi;
        float wj = inv_masses[j];

        float len = length(d);

        if (wi + wj == 0.0 || len == 0.0) {
            continue;
        }

        correction += wi / (wi + wj) *
                      (len - l) * (d / len);

        count++;
    }

    if (count > 0) {
        correction /= float(count);
    }

    // Apply spring correction.
    float3 x = xi + correction;

    float4 plane = contact_planes[i];

    float3 n = plane.xyz;
    float plane_d = plane.w;

    // A zero normal means no collision.
    if (dot(n, n) > 0.0) {

        // Calculate penetration depth.
        float penetration = max(
            0.0,
            plane_d - dot(n, x)
        );

        // Move particle outside contact plane.
        x += penetration * n;

        float3 displacement = x - old_positions[i];

        // Remove normal component of displacement.
        float3 tangent = displacement -
                         dot(displacement, n) * n;

        float tangent_len = length(tangent);

        if (tangent_len <= static_friction * penetration) {

            // Static friction:
            // Completely eliminate tangential movement.
            x -= tangent;
        }
        else if (tangent_len > 0.0) {

            // Kinetic friction:
            // Reduce tangential movement.
            x -= (kinetic_friction * penetration / tangent_len)
                 * tangent;
        }
    }

    positions_out[i] = x;
}


[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     uniform float dt) {

    if (tid >= num_particles) {
        return;
    }

    // Static particles have zero velocity.
    if (inv_masses[tid] == 0.0) {
        velocities[tid] = float3(0.0, 0.0, 0.0);
        return;
    }

    // Calculate velocity from corrected position.
    float3 velocity = (
        positions[tid] - old_positions[tid]
    ) / dt;

    velocities[tid] = velocity;
}
