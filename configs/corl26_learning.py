from optcar.models.fkd.learning_config import SupervisedLearningConfig

def learning_config() -> SupervisedLearningConfig:
    return SupervisedLearningConfig(
        epochs=500,
        batch_size=4096,
        
        lr=1e-3,
        adamW_weight_decay=1e-2,
        discount_gamma=1.0,

        T_MSE_factor=0.005,
        T_DELTA_MSE_factor=1.0,
        T_DELTA_MAE_factor_y = 0.0,
        R_Geodesic_factor=0.0,
        R_DELTA_Geodesic_factor=0.3,
        T_DELTA_Vel_MSE_factor=0.0,
        

        use_lr_warmup=True,
        lr_warmup_factor=0.01,
        use_cosine_annealing=True,
        use_gradient_clip=False,
        gradient_clip_val=0.001,

        save_checkpoint_every_n_epoch=1,

        checkpoint_folder = "/scratch/rwik/fkd_chkpts",
    )
