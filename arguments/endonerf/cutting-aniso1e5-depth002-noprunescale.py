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
    densify_until_iter = 5000,
    prune_until_iter = 6000, # avoid floaters
    prune_scale_extent_ratio = -1,       # was 10.0 (cutting-depthreg-aniso1e5-depth002) -- cutting's real value (disabled); ablation isolating whether this floater-ratio pruning criterion (vs prune_until_iter, or anisotropy) is what causes the small "orig" regression
    position_lr_max_steps = 6000,
    pruning_interval = 100,
    sideview_smooth_weight = 0,          # cutting's real value -- color TV loss off
    sideview_depth_weight = 0.02,        # matches cutting-depthreg-aniso1e5-depth002 -- best-found depth_weight
    sideview_depth_huber_beta = 1.0,     # matches the new default (see debugtools/depth_check.py); explicit here for reproducibility
    sideview_reg_interval = 5,           # matches cutting-sidereg's tuning, not cutting's own (inert, unused there) value of 10
    sideview_elev = 20,                  # matches cutting
    sideview_azims = -1,                 # matches cutting-sidereg's tuning, not cutting's own (inert, unused there) value of [0,90,180,270]
    anisotropy_weight = 1e-5,             # best-found value from the anisotropy sweep (5e-5 too strong, 2e-6 too weak)
    anisotropy_ratio_power = 2.0,         # matches cutting-quadanireg -- quadratic penalty on (ratio - 1)
    anisotropy_ratio_threshold = 10.0,    # matches cutting-quadanireg
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
