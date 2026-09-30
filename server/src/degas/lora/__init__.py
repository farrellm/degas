"""`degas lora`: train an SDXL character LoRA on a Colab GPU with kohya sd-scripts.

Not part of the app (design §1 lists training as a non-goal): a CLI that reuses the Colab
CLI wrapper, SSH tunnel and Drive token, in its own Colab session so it never touches the
inference session. The dataset is a local folder of images with `.txt` captions and a
`lora.toml`; runs, checkpoints and samples go to `<data_dir>/lora-runs/<run>/`.
"""
