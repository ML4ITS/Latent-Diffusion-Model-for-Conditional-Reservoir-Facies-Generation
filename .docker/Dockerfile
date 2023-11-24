FROM python:3.9

# set USER
RUN useradd -m -r -u 1101 newuser  # create user
RUN python3 -m pip install --upgrade pip
USER newuser

# install pacakges using the virtualenv's pip
RUN python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN python3 -m pip install denoising_diffusion_pytorch
RUN python3 -m pip install einops numpy matplotlib scikit-learn wandb pandas
RUN python3 -m pip install pytorch-lightning lion_pytorch
#RUN python3 -m pip install
#RUN python3 -m pip install


# to enter into the command window (instead of python window)
CMD ["/bin/bash"]