import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


# --- Generator Definition ---
class Generator(nn.Module):
    def __init__(
        self, in_channels: int = 1, out_channels: int = 1, dropout_prob: float = 0.5
    ):
        super().__init__()
        # --- Encoder ---
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=4, stride=2, padding=1
        )  # 28→14
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 14→7
        self.bn2 = nn.BatchNorm2d(128)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # --- Bottleneck ---
        self.bottleneck_conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bottleneck_bn = nn.BatchNorm2d(256)
        self.bottleneck_act = nn.ReLU(inplace=True)

        # --- Decoder block 1 (7→14) ---
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # after cat with e1 (64 channels): 128 + 64 = 192 → reduce to 128
        self.conv3 = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.drop1 = nn.Dropout2d(dropout_prob)

        # --- Decoder block 2 (14→28) ---
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # after cat with x (in_channels=1): 64 + 1 = 65 → reduce to 64
        self.conv4 = nn.Conv2d(64 + in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(dropout_prob)

        # --- Final projection ---
        self.final = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        e1 = self.act(self.bn1(self.conv1(x)))  # [B, 64,14,14]
        e2 = self.act(self.bn2(self.conv2(e1)))  # [B,128, 7, 7]

        # Bottleneck
        b = self.bottleneck_act(
            self.bottleneck_bn(self.bottleneck_conv(e2))
        )  # [B,256,7,7]

        # Decoder block 1
        d1 = self.act(self.bn3(self.up1(b)))  # [B,128,14,14]
        d1 = torch.cat([d1, e1], dim=1)  # [B,192,14,14]
        d1 = self.act(self.bn3b(self.conv3(d1)))  # [B,128,14,14]
        d1 = self.drop1(d1)

        # Decoder block 2
        d2 = self.act(self.bn4(self.up2(d1)))  # [B, 64,28,28]
        d2 = torch.cat([d2, x], dim=1)  # [B, 65,28,28]
        d2 = self.act(self.bn4b(self.conv4(d2)))  # [B, 64,28,28]
        d2 = self.drop2(d2)

        # Output
        return self.tanh(self.final(d2))  # [B,  1,28,28]


# --- Load model checkpoint ---
@st.cache_resource
def load_generator(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(checkpoint_path, map_location=device))
    gen.eval()
    return gen, device


# --- Load MNIST test dataset ---
@st.cache_data
def load_mnist():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    # group indices by label
    label_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        label_indices[label].append(idx)
    return dataset, label_indices


# --- App UI ---
st.title("Pix2Pix MNIST Generator")
checkpoint_path = st.text_input("Checkpoint path", "g_epoch_7.pth")
num = st.selectbox("Select digit to condition on", list(range(10)))

if st.button("Generate Samples"):
    gen, device = load_generator(checkpoint_path)
    dataset, label_indices = load_mnist()
    # pick a random sample for the chosen digit
    idx = np.random.choice(label_indices[num])
    img, _ = dataset[idx]
    x = img.unsqueeze(0).to(device)  # [1,1,28,28]
    # generate 5 samples
    x_cond = x.repeat(5, 1, 1, 1)
    noise = torch.randn_like(x_cond) * 0.1
    with torch.no_grad():
        gen_imgs = gen(x_cond + noise)
    # denormalize and convert to numpy images
    imgs_out = (gen_imgs.cpu() * 0.5 + 0.5).squeeze(1).numpy()
    # display
    cols = st.columns(5)
    for i, col in enumerate(cols):
        arr = (imgs_out[i] * 255).astype(np.uint8)
        img_disp = Image.fromarray(arr, mode="L")
        col.image(img_disp, caption=f"Sample {i + 1}", use_column_width=True)

# EOF
