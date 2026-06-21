# pix2pix-app

A Streamlit web application that demonstrates a pix2pix-inspired conditional image generation model trained on the MNIST handwritten digit dataset. Given a real MNIST digit image as a conditioning input, the generator produces new variations of that digit using a U-Net-style architecture.

## What is pix2pix?

Pix2pix is a conditional Generative Adversarial Network (cGAN) framework for image-to-image translation. Instead of generating images from random noise alone, it conditions the generator on an input image and learns a mapping from that input to a target output. In this project, the model is trained on MNIST digits: a real digit image is used as the condition, and the generator learns to synthesize plausible variations of the same digit class.

## What this app does

- Loads a pre-trained generator checkpoint (`.pth` file).
- Downloads the MNIST test set automatically on first run.
- Lets you select any digit (0-9) from a dropdown.
- Picks a random real MNIST image of that digit as the conditioning input.
- Feeds the conditioned input (with small additive Gaussian noise) into the generator five times in parallel to produce five distinct output samples.
- Displays all five generated 28x28 grayscale images side by side in the browser.

## Tech stack

| Component | Technology |
|-----------|-----------|
| Deep learning framework | PyTorch |
| Web app framework | Streamlit |
| Image processing | Pillow, NumPy |
| Dataset | MNIST (via torchvision) |
| Pre-trained checkpoints | `g_epoch_7.pth`, `g_epoch_9.pth` |

## Architecture overview

The generator is a compact U-Net with skip connections operating on 28x28 grayscale images.

**Encoder**

- `conv1`: Conv2d(1, 64, 4x4, stride=2) — 28x28 -> 14x14
- `conv2`: Conv2d(64, 128, 4x4, stride=2) — 14x14 -> 7x7
- Activation: LeakyReLU(0.2) with BatchNorm2d

**Bottleneck**

- `bottleneck_conv`: Conv2d(128, 256, 3x3, stride=1) — 7x7, ReLU + BatchNorm2d

**Decoder with skip connections**

- `up1`: ConvTranspose2d(256, 128, 4x4, stride=2) — 7x7 -> 14x14, concatenated with encoder output `e1` (192 channels total), then Conv2d(192, 128) + Dropout2d
- `up2`: ConvTranspose2d(128, 64, 4x4, stride=2) — 14x14 -> 28x28, concatenated with the original input `x` (65 channels total), then Conv2d(65, 64) + Dropout2d
- `final`: Conv2d(64, 1, 3x3) + Tanh activation

Output pixel values are in the range [-1, 1] and are denormalized to [0, 255] for display.

The skip connections (U-Net style) allow low-level spatial features from the encoder to pass directly to the decoder, helping preserve structural detail of the conditioning digit.

## Project structure

```
pix2pix-app/
├── app.py                  # Streamlit app: model definition, loading, and UI
├── g_epoch_7.pth           # Generator checkpoint saved at epoch 7
├── g_epoch_9.pth           # Generator checkpoint saved at epoch 9
├── requirements.txt        # Python dependencies
├── data/
│   └── MNIST/              # MNIST dataset (auto-downloaded on first run)
└── .devcontainer/
    └── devcontainer.json   # GitHub Codespaces / VS Code dev container config
```

## Installation and setup

**Prerequisites:** Python 3.9 or later.

1. Clone the repository:

   ```bash
   git clone https://github.com/YashNagraj75/pix2pix-app.git
   cd pix2pix-app
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   This installs: `streamlit`, `torch`, `torchvision`, `pillow`, `numpy`.

3. Ensure a checkpoint file is present in the project root. Both `g_epoch_7.pth` and `g_epoch_9.pth` are included in the repository.

## Usage

Start the Streamlit server:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

In the UI:

1. **Checkpoint path** — enter `g_epoch_7.pth` or `g_epoch_9.pth` (default is `g_epoch_7.pth`).
2. **Select digit** — choose which digit class (0-9) to condition on.
3. Click **Generate Samples** — five generated images appear side by side.

The MNIST test set (~10 MB) is downloaded automatically to `./data/` on the first run if it is not already present.

## Running in GitHub Codespaces

The repository includes a `.devcontainer` configuration. Opening the repo in a Codespace will automatically:

- Install all Python dependencies.
- Start the Streamlit app on port 8501.
- Forward port 8501 and open a preview in the browser.

No additional setup steps are required in a Codespace.

## Notes

- GPU acceleration is used automatically when a CUDA-capable device is available; otherwise the model runs on CPU.
- The model is cached by Streamlit (`@st.cache_resource`) so the checkpoint is only loaded once per session.
- The MNIST dataset is also cached (`@st.cache_data`) to avoid reloading on each interaction.
- Dropout (p=0.5) is applied during inference as well, which introduces stochasticity and contributes to the variation across the five generated samples.
