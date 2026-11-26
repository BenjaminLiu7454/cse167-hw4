struct BVHNode {
    float3 bbox_min;
    float3 bbox_max;


    // which BVH node we should skip
    // to if this node is totally not
    // intersecting the ray
    // if next_index < 0, then we don't
    // have the next node to skip to
    int next_index;


    uint triangle_begin;
    uint triangle_end;


    bool is_leaf;
};


float point_aabb_distance(float3 x, float3 bmin, float3 bmax) {
    return length(x - clamp(x, bmin, bmax));
}


struct ClosestPoint {
    float3 position;
    float3 bary;
};


// From https://iquilezles.org/articles/distfunctions/
ClosestPoint closest_point_triangle(
        float3 p, float3 a, float3 b, float3 c) {
    float3 ba = b - a, pa = p - a;
    float3 cb = c - b, pb = p - b;
    float3 ac = a - c, pc = p - c;
    float3 nor = cross(ba, ac);


    float t0 = clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0);
    float t1 = clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0);
    float t2 = clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0);


    float3 q0 = a + ba * t0;
    float3 q1 = b + cb * t1;
    float3 q2 = c + ac * t2;


    if (sign(dot(cross(ba, nor), pa)) +
        sign(dot(cross(cb, nor), pb)) +
        sign(dot(cross(ac, nor), pc)) < 2.0) {


        float d0 = dot(p - q0, p - q0);
        float d1 = dot(p - q1, p - q1);
        float d2 = dot(p - q2, p - q2);


        if (d0 < d1 && d0 < d2)
            return {q0, float3(1.0 - t0, t0, 0.0)};
        if (d1 < d2)
            return {q1, float3(0.0, 1.0 - t1, t1)};
        return {q2, float3(t2, 0.0, 1.0 - t2)};
    }


    float inv_nor2 = 1.0 / dot(nor, nor);
    float3 q = p - nor * (dot(nor, pa) * inv_nor2);


    float wb = dot(cross(pa, ac), nor) * inv_nor2;
    float wc = dot(cross(pa, ba), nor) * inv_nor2;
    return {q, float3(1.0 - wb - wc, wb, wc)};
}



[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      StructuredBuffer<float4> planes,
                      StructuredBuffer<float4> spheres,
                      StructuredBuffer<float3> mesh_vertices,
                      StructuredBuffer<float3> mesh_vertex_normals,
                      StructuredBuffer<uint3> mesh_triangles,
                      StructuredBuffer<BVHNode> bvh_nodes,
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      uniform uint num_bvh_nodes,
                      uniform float collision_distance,
                      uniform float contact_margin) {

    if (tid >= num_particles) {
        return;
    }

    // Initialize contact plane to no collision.
    contact_planes[tid] = float4(0.0, 0.0, 0.0, 0.0);

    if (inv_masses[tid] == 0.0) {
        return;
    }

    float3 x = positions[tid];

    // Maximum distance at which we consider a collision.
    float contact_threshold = contact_margin + collision_distance;

    // Track the closest signed distance across all colliders.
    float min_distance = contact_threshold;

    float4 best_contact = float4(0.0, 0.0, 0.0, 0.0);

    for (uint i = 0; i < num_planes; i++) {

        float4 plane = planes[i];

        float3 n = plane.xyz;
        float d = plane.w;

        // Signed distance to plane.
        float signed_distance = dot(n, x) - d;

        if (signed_distance < min_distance) {

            min_distance = signed_distance;

            // Shift contact plane outward.
            best_contact = float4(
                n,
                d + collision_distance
            );
        }
    }

    for (uint i = 0; i < num_spheres; i++) {

        float4 sphere = spheres[i];

        float3 center = sphere.xyz;
        float radius = sphere.w;

        float3 direction = x - center;

        float distance_from_center = length(direction);

        // Signed distance to sphere surface.
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

            // Find closest point on sphere surface.
            float3 surface_point = center + radius * n;

            float d = dot(n, surface_point);

            best_contact = float4(
                n,
                d + collision_distance
            );
        }
    }

    // Start with the maximum contact distance.
    float mesh_min_distance = contact_threshold;

    float4 mesh_contact = float4(0.0, 0.0, 0.0, 0.0);

    // Start stackless traversal at the root.
    uint node_index = 0;

    while (node_index < num_bvh_nodes) {

        BVHNode node = bvh_nodes[node_index];

        // Calculate distance from particle to bounding box.
        float box_distance = point_aabb_distance(
            x,
            node.bbox_min,
            node.bbox_max
        );

        if (box_distance > mesh_min_distance) {

            if (node.next_index < 0) {
                break;
            }

            node_index = uint(node.next_index);

            continue;
        }

        if (node.is_leaf) {

            for (uint j = node.triangle_begin;
                 j < node.triangle_end;
                 j++) {

                // Get triangle vertex indices.
                uint3 tri = mesh_triangles[j];

                // Get triangle vertices.
                float3 p0 = mesh_vertices[tri.x];
                float3 p1 = mesh_vertices[tri.y];
                float3 p2 = mesh_vertices[tri.z];

                // Find closest point on triangle.
                ClosestPoint closest = closest_point_triangle(x, p0, p1, p2);

                float3 q = closest.position;

                float distance = length(x - q);

                // Only process triangles closer than
                // the current closest triangle.
                if (distance < mesh_min_distance) {

                    mesh_min_distance = distance;

                    float3 ng = normalize(
                        cross(p1 - p0, p2 - p0)
                    );

                    float3 n0 = mesh_vertex_normals[tri.x];
                    float3 n1 = mesh_vertex_normals[tri.y];
                    float3 n2 = mesh_vertex_normals[tri.z];

                    float3 ns = normalize(
                        closest.bary.x * n0 +
                        closest.bary.y * n1 +
                        closest.bary.z * n2
                    );

                    if (dot(ng, ns) < 0.0) {
                        ng = -ng;
                    }

                    float signed_distance = dot(ng, x - q);

                    float d = dot(ng, q);

                    float4 candidate_contact = float4(
                        ng,
                        d + collision_distance
                    );

                    // Compare against all other colliders.
                    if (signed_distance < min_distance) {

                        min_distance = signed_distance;

                        best_contact = candidate_contact;
                    }
                }
            }
        }

        node_index++;
    }

    if (min_distance < contact_threshold) {
        contact_planes[tid] = best_contact;
    }
    else {
        contact_planes[tid] = float4(
            0.0, 0.0, 0.0, 0.0
        );
    }
}
