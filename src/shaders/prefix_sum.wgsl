#import "embedded://strand_software_rasterizer/shaders/types.wgsl"::{
    PushConstants,
}

const SCAN_THREADS: u32 = #{NUMBER_OF_THREADS_PER_WORKGROUP}; // e.g., 256
const SCAN_SUBGROUP_THREADS: u32 = #{NUMBER_OF_THREADS_PER_SUBGROUP}; // e.g., 32
const SCAN_SUBGROUPS: u32 = SCAN_THREADS / SCAN_SUBGROUP_THREADS; // e.g., 8

// Shared memory for inter-subgroup communication and temporary storage
var<workgroup> subgroup_partials: array<u32, SCAN_SUBGROUPS>;
var<workgroup> wg_scan_storage: array<u32, SCAN_THREADS>; // For Blelloch intermediate/final values

fn get_scan_workgroup_index(workgroup_id: vec3u, num_workgroups: vec3u, pc: PushConstants) -> u32 {
    let flat_grid_index = workgroup_id.y * num_workgroups.x + workgroup_id.x;
    return flat_grid_index + pc.workgroup_offset;
}

// Performs an INCLUSIVE scan within the workgroup using Blelloch method modification.
// Stores the intermediate (potentially zeroed last element) result in wg_scan_storage.
// Returns the total sum of the workgroup (only valid for the last thread).
fn workgroup_inclusive_scan_blelloch(
    local_id: vec3u,
    subgroup_id: u32,
    subgroup_local_id: u32,
    value: u32
) -> u32 {
    let sg_inclusive_sum = subgroupInclusiveAdd(value);

    // Last lane of each subgroup writes its inclusive sum (subgroup total) to shared memory
    if local_id.x < SCAN_SUBGROUPS { subgroup_partials[local_id.x] = 0u; }
    workgroupBarrier();
    if subgroup_local_id == SCAN_SUBGROUP_THREADS - 1u {
        subgroup_partials[subgroup_id] = sg_inclusive_sum;
    }
    workgroupBarrier();

    // Scan the subgroup partial sums (sequentially by thread 0) -> Exclusive Prefix
    if local_id.x == 0u {
        var accumulator = 0u;
        for (var i = 0u; i < SCAN_SUBGROUPS; i = i + 1u) {
            let temp = subgroup_partials[i];
            subgroup_partials[i] = accumulator; // Write exclusive prefix
            accumulator += temp;
        }
    }
    workgroupBarrier();

    // Combine subgroup scan results with the scanned partials (exclusive prefix)
    let prefix_for_my_subgroup = subgroup_partials[subgroup_id];
    let block_local_inclusive_sum = sg_inclusive_sum + prefix_for_my_subgroup;

    // Store intermediate result in workgroup storage (for Blelloch down-sweep prep)
    wg_scan_storage[local_id.x] = block_local_inclusive_sum;
    workgroupBarrier();

    // Blelloch modification: Last thread reads total sum AND zeros its storage slot
    var total_wg_sum = 0u;
    if local_id.x == SCAN_THREADS - 1u {
        total_wg_sum = wg_scan_storage[local_id.x]; // Read total sum before zeroing
        wg_scan_storage[local_id.x] = 0u;          // Zero last element's storage
    }
    // Note: total_wg_sum only valid on last thread here.
    workgroupBarrier();

    // Intermediate results for down-sweep are now in wg_scan_storage (last element zeroed)
    return total_wg_sum;
}

// Performs an EXCLUSIVE scan within the workgroup.
// Stores the final exclusive scan result in wg_scan_storage.
// Returns the total sum of the workgroup (exclusive sum of last element + value of last element).
// Only valid for the last thread.
fn workgroup_exclusive_scan(
    local_id: vec3u,
    subgroup_id: u32,
    subgroup_local_id: u32,
    value: u32
) -> u32 {

    let sg_exclusive_sum = subgroupExclusiveAdd(value);
    let sg_inclusive_sum = subgroupInclusiveAdd(value); // Need inclusive sum for totals

    // Last lane of each subgroup writes its INCLUSIVE sum (subgroup total) to shared memory
    if local_id.x < SCAN_SUBGROUPS { subgroup_partials[local_id.x] = 0u; }
    workgroupBarrier();
    if subgroup_local_id == SCAN_SUBGROUP_THREADS - 1u {
        subgroup_partials[subgroup_id] = sg_inclusive_sum;
    }
    workgroupBarrier();

    // Scan the subgroup partial sums (sequentially by thread 0) -> Exclusive Prefix
    if local_id.x == 0u {
        var accumulator = 0u;
        for (var i = 0u; i < SCAN_SUBGROUPS; i = i + 1u) {
            let temp = subgroup_partials[i]; // Read original inclusive sum
            subgroup_partials[i] = accumulator; // Write exclusive prefix
            accumulator += temp;
        }
    }
    workgroupBarrier();

    // Combine subgroup exclusive scan results with the scanned partials (exclusive prefix)
    let prefix_for_my_subgroup = subgroup_partials[subgroup_id];
    let final_exclusive_sum = sg_exclusive_sum + prefix_for_my_subgroup;

    // Store final exclusive sum in workgroup storage
    wg_scan_storage[local_id.x] = final_exclusive_sum;
    workgroupBarrier();

    // Calculate total sum (last element's exclusive sum + last element's value)
    var total_wg_sum = 0u;
    if local_id.x == SCAN_THREADS - 1u {
        // Read the last element's computed exclusive sum and add its original value
        total_wg_sum = wg_scan_storage[SCAN_THREADS - 1u] + value;
    }
    // Return value only used by last thread.
    return total_wg_sum;
}
