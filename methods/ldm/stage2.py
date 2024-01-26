from methods.ldm.utils import diffusion_trainer_fn

# train the diffusion model
trainer = diffusion_trainer_fn(initialize_wandb=True)
trainer.train()
