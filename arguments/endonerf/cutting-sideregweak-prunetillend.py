ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10
)

OptimizationParams = dict(
    coarse_iterations = 1000,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    iterations = 6000,
    percent_dense = 0.01,
    prune_scale_extent_ratio = 10.0,
    densify_until_iter = 5000,
    prune_until_iter = 6000,
    position_lr_max_steps = 6000,
    pruning_interval = 100,
    sideview_smooth_weight = 0.001,
    sideview_reg_interval = 5,   # was 10 -- regularize side views twice as often to fight remaining holes
    sideview_elev = 20,
    sideview_azims = -1,
    anisotropy_weight = 0,  # no anisotropy regularization
    anisotropy_ratio_threshold = 10.0,
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
     'resolution': [64, 64, 64, 100]
    },
    multires = [1, 2, 4, 8],
    defor_depth = 0,
    net_width = 32,
    plane_tv_weight = 2e-2,
    time_smoothness_weight = 2e-2,
    l1_time_planes =  2e-2,
    weight_decay_iteration=0,
)
