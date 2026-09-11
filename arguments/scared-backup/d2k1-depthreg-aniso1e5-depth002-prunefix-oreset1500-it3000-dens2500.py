# RESULT: worse than baseline at every elevation, and very unstable (elev5 seeds 20.26-28.35).
# Capping densification did cut points 61k -> 44k as intended, but quality collapsed with it.

ModelParams = dict(
    extra_mark = 'scared',
    no_fine=False,
    init_pts=20_000
)

OptimizationParams = dict(
    coarse_iterations = 1000,
    iterations = 3000,                   # 2x the reset interval, so the final reset lands on the save iteration and costs nothing
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 3000,
    densify_until_iter = 2500,           # stop adding gaussians 500 iters before the end, as endonerf does (5000 of 6000)
    prune_until_iter = 3000,             # keep pruning to the end; -1 would tie it to densify_until_iter and stop early

    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,

    pruning_interval = 100,              # was 2000, which with iterations=2000 pruned about once
    prune_scale_extent_ratio = 10.0,     # was -1 (off) -- scale.max() > 10 * extent(16.13) = 161 pruned as floater
    percent_dense = 0.01,
    opacity_reset_interval = 1500,       # does not divide coarse(1000) so coarse never resets; one real reset at fine 1500 with 1500 iters to recover

    # depth reg + anisotropy reg + endonerf-style floater pruning, matching endonerf/cutting-depthreg-aniso1e5-depth002. The whole
    # block is copied rather than just the two weights that differ from the endonerf baseline,
    # since sideview_reg_interval defaults to -1 (off) and the shape params default differently
    sideview_smooth_weight = 0,          # color TV loss off, as on cutting
    sideview_depth_weight = 0.02,
    sideview_depth_huber_beta = 1.0,
    sideview_reg_interval = 5,
    sideview_elev = 20,
    sideview_azims = -1,                 # sample azimuth uniformly from [0, 360)
    anisotropy_weight = 1e-5,
    anisotropy_ratio_power = 2.0,
    anisotropy_ratio_threshold = 10.0,
)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 100]
    },
    multires = [1, 2, 4, 8],
    defor_depth = 0,
    net_width = 32,
    plane_tv_weight = 0,
    time_smoothness_weight = 0,
    l1_time_planes =  0,
    weight_decay_iteration=0,
    no_dx=False,
    no_ds=True,
    no_dr=True,
    no_do=False
)

RuntimeParams = dict(
    log_point_counts = True,
)
