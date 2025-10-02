
use bevy::render::render_resource::ComputePass;


pub fn dispatch_workgroup_ext(
    pass: &mut ComputePass,
    number_of_workgroups: u32,
    max_compute_workgroups_per_dimension: u32,
    workgroup_offset_offset: u32,
) {
    pass.set_push_constants(workgroup_offset_offset, bytemuck::bytes_of(&0));

    if number_of_workgroups <= max_compute_workgroups_per_dimension {
        pass.dispatch_workgroups(number_of_workgroups, 1, 1);
    } else {
        let d = number_of_workgroups / max_compute_workgroups_per_dimension;

        pass.dispatch_workgroups(max_compute_workgroups_per_dimension, d, 1);

        let workgroup_offset = max_compute_workgroups_per_dimension * d;
        pass.set_push_constants(
            workgroup_offset_offset,
            bytemuck::bytes_of(&workgroup_offset),
        );
        pass.dispatch_workgroups(number_of_workgroups - workgroup_offset, 1, 1);
    }
}