"""X-CLIP guided watermark decoder (Sec. 4.2, Supp. A.1 and B).

Stage 1 trains a lightweight MLP D_M on top of the *frozen* X-CLIP text encoder E_W:
an L-bit message is mapped to an X-CLIP token sequence W (Eq. 8, Supp. Alg. 1),
encoded by E_W, and decoded back to the message (Eq. 9) with a BCE loss (Eq. 10).

Stage 2 keeps D_M frozen and decodes rendered clips through the *frozen* X-CLIP
video encoder E_V instead (Eq. 11).
"""

import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import XCLIPModel

XCLIP_VERSION = "microsoft/xclip-base-patch32"  # ViT-B/32 X-CLIP, 512-d latent space
XCLIP_NUM_FRAMES = 8                             # T = 8 frames per clip
XCLIP_EMBED_DIM = 512

# CLIP BPE vocabulary (|V| = 49,408): 0 = padding, 49406 = <start>, 49407 = <end>.
XCLIP_TOKEN_PAD = 0
XCLIP_TOKEN_MIN = 1
XCLIP_TOKEN_MAX = 49405
XCLIP_TOKEN_START = 49406
XCLIP_TOKEN_END = 49407
XCLIP_TOKEN_LEN = 77  # W_max

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_IMAGE_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Bit-to-token mapping (Eq. 8, Supp. Alg. 1)
# ---------------------------------------------------------------------------
def build_bit_to_token_map(msg_len: int, generator: torch.Generator = None) -> torch.Tensor:
    """Phi: randomly assign two distinct valid token IDs to every bit position.

    Returns a (L, 2) LongTensor where Phi[i, b] is the token of bit value b at position i.
    The mapping is sampled once and kept fixed for training and evaluation.
    """
    num_valid = XCLIP_TOKEN_MAX - XCLIP_TOKEN_MIN + 1
    if 2 * msg_len > num_valid:
        raise ValueError(f"msg_len={msg_len} is too long for the X-CLIP vocabulary.")
    perm = torch.randperm(num_valid, generator=generator)[: 2 * msg_len] + XCLIP_TOKEN_MIN
    return perm.reshape(msg_len, 2).long()


def message_to_tokens(message: torch.Tensor, b2t_map: torch.Tensor) -> torch.Tensor:
    """W = {w_start} U {Phi(m_i, i)}_i U {w_end}, zero-padded to W_max (Eq. 8)."""
    msg_len = b2t_map.shape[0]
    if msg_len + 2 > XCLIP_TOKEN_LEN:
        raise ValueError(f"msg_len={msg_len} does not fit into {XCLIP_TOKEN_LEN} tokens.")
    tokens = torch.full((XCLIP_TOKEN_LEN,), XCLIP_TOKEN_PAD, dtype=torch.long)
    tokens[0] = XCLIP_TOKEN_START
    tokens[1 : msg_len + 1] = b2t_map[torch.arange(msg_len), message.long()]
    tokens[msg_len + 1] = XCLIP_TOKEN_END
    return tokens


def sample_messages(num: int, msg_len: int, rng: random.Random) -> torch.Tensor:
    """Sample `num` distinct random L-bit messages, returned as a (num, L) float tensor."""
    num = min(num, 2 ** msg_len)
    seen, messages = set(), []
    while len(messages) < num:
        value = rng.getrandbits(msg_len)
        if value in seen:
            continue
        seen.add(value)
        messages.append([int(b) for b in format(value, f"0{msg_len}b")])
    return torch.tensor(messages, dtype=torch.float32)


class MessageTokenDataset(Dataset):
    """(token sequence, binary message) pairs for Stage-1 decoder training."""

    def __init__(self, messages: torch.Tensor, b2t_map: torch.Tensor):
        self.messages = messages
        self.tokens = torch.stack([message_to_tokens(m, b2t_map) for m in messages])

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        return self.tokens[idx], self.messages[idx]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def build_message_decoder(msg_len: int) -> nn.Module:
    """D_M: 3-layer MLP, 512 -> 512 -> 256 -> L (Supp. B).

    The final Sigmoid of the paper is folded into BCEWithLogits, so D_M outputs logits.
    """
    return nn.Sequential(
        nn.Linear(XCLIP_EMBED_DIM, 512),
        nn.GELU(),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Linear(256, msg_len),
    )


def preprocess_video(frames: torch.Tensor) -> torch.Tensor:
    """(T, 3, H, W) in [0, 1] -> (1, T, 3, 224, 224) CLIP-normalized, differentiable."""
    if frames.dim() == 4:
        frames = frames.unsqueeze(0)
    b, t, c, h, w = frames.shape
    x = F.interpolate(frames.reshape(b * t, c, h, w), size=CLIP_IMAGE_SIZE, mode="bilinear", align_corners=False)
    x = x.reshape(b, t, c, *CLIP_IMAGE_SIZE)
    mean = torch.tensor(CLIP_IMAGE_MEAN, device=x.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD, device=x.device).view(1, 1, 3, 1, 1)
    return (x - mean) / std


class XCLIPMessageDecoder(nn.Module):
    """Frozen X-CLIP encoders (E_W, E_V) followed by the trainable message decoder D_M."""

    def __init__(self, msg_len: int, xclip_version: str = XCLIP_VERSION):
        super().__init__()
        self.msg_len = msg_len
        self.xclip = XCLIPModel.from_pretrained(xclip_version)
        self.xclip.eval()
        for p in self.xclip.parameters():
            p.requires_grad_(False)
        self.msg_decoder = build_message_decoder(msg_len)

    def train(self, mode: bool = True):
        super().train(mode)
        self.xclip.eval()  # X-CLIP always stays frozen
        return self

    @staticmethod
    def _normalize(feats: torch.Tensor) -> torch.Tensor:
        return feats / feats.norm(dim=-1, keepdim=True)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Eq. (9): M_hat = D_M(E_W(W)). tokens: (B, 77) -> logits (B, L)."""
        attention_mask = (tokens != XCLIP_TOKEN_PAD).long()
        feats = self.xclip.get_text_features(input_ids=tokens, attention_mask=attention_mask)
        return self.msg_decoder(self._normalize(feats.float()))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Eq. (11): M_hat = D_M(E_V(I_S)). frames: (T, 3, H, W) in [0, 1] -> logits (1, L)."""
        feats = self.xclip.get_video_features(pixel_values=preprocess_video(frames))
        return self.msg_decoder(self._normalize(feats.float()))


def save_decoder(path: str, model: XCLIPMessageDecoder, b2t_map: torch.Tensor, extra: dict = None):
    torch.save(
        {
            "msg_len": model.msg_len,
            "msg_decoder": model.msg_decoder.state_dict(),
            "b2t_map": b2t_map.cpu(),
            **(extra or {}),
        },
        path,
    )


def load_decoder(path: str, device: str = "cuda") -> tuple:
    """Load a Stage-1 checkpoint and freeze it. Returns (decoder, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu")
    model = XCLIPMessageDecoder(ckpt["msg_len"])
    model.msg_decoder.load_state_dict(ckpt["msg_decoder"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


@torch.no_grad()
def bit_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Percentage of correctly recovered bits (logit > 0 -> bit 1)."""
    pred = (logits > 0).float()
    target = target.expand_as(pred)
    return (pred == target).float().mean().item() * 100.0
