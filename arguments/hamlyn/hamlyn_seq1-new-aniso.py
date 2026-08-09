ModelParams = dict(
    extra_mark = 'hamlyn',
    camera_extent = 9
)

OptimizationParams = dict(
    coarse_iterations = 3000,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.00016,
    iterations = 6000,
    percent_dense = 0.01,
    prune_scale_extent_ratio = 0.5,
    densify_until_iter = 5000,
    position_lr_max_steps = 5000,
    pruning_interval = 100,
    sideview_smooth_weight = 0.08,
    sideview_reg_interval = 10,
    anisotropy_weight = 0.01,  # was 0 -- never trained on Hamlyn; matches EndoNeRF's value
    anisotropy_ratio_threshold = 5.0,
    anisotropy_size_power = 0.5
)

TestEvalParams = dict(
    test_eval_interval = 500,
    test_eval_start_iter = 0,
    test_eval_max_views = 0,
)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 64,
     'resolution': [64, 64, 64, 100] # x2

    },
    multires = [1, 2, 4, 8],
    defor_depth = 0,
    net_width = 32,
    plane_tv_weight = 2e-2,
    time_smoothness_weight = 2e-2,
    l1_time_planes =  2e-2,
    weight_decay_iteration=0,
)
