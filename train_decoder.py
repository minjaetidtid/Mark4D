"""Stage 1: train the X-CLIP guided message decoder D_M (Sec. 4.2, Supp. A.1 and B).

Example:
    python train_decoder.py --msg_len 48 --out_dir checkpoints/decoder \
        --num_epochs <E> --batch_size <B> --lr <LR>
"""

import argparse
import os
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import trange

from mark4d.decoder import (
    MessageTokenDataset,
    XCLIPMessageDecoder,
    bit_accuracy,
    build_bit_to_token_map,
    sample_messages,
    save_decoder,
)
from mark4d.utils import seed_everything


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits, targets = [], []
    for tokens, msgs in loader:
        logits.append(model.decode_tokens(tokens.cuda()))
        targets.append(msgs.cuda())
    return bit_accuracy(torch.cat(logits), torch.cat(targets))


def main(args):
    seed_everything(args.seed)

    # Phi is sampled once and kept fixed afterwards (Supp. A.1).
    b2t_map = build_bit_to_token_map(args.msg_len, torch.Generator().manual_seed(args.seed))

    # 5,000 random binary messages, 80:20 train/test split (Supp. B).
    messages = sample_messages(args.num_messages, args.msg_len, random.Random(args.seed))
    dataset = MessageTokenDataset(messages, b2t_map)
    perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_train = int(0.8 * len(dataset))
    train_set, test_set = Subset(dataset, perm[:n_train]), Subset(dataset, perm[n_train:])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = XCLIPMessageDecoder(args.msg_len).cuda()
    optimizer = optim.Adam(model.msg_decoder.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()  # Eq. (10)

    pbar = trange(1, args.num_epochs + 1, desc="Decoder")
    for epoch in pbar:
        model.train()
        total = 0.0
        for tokens, msgs in train_loader:
            loss = criterion(model.decode_tokens(tokens.cuda()), msgs.cuda())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(msgs)
        test_acc = evaluate(model, test_loader)
        pbar.set_description(f"[{epoch}] loss {total / len(train_set):.4f} | test bit acc {test_acc:.2f}%")

    train_acc = evaluate(model, train_loader)
    print(f"Final bit accuracy: train {train_acc:.2f}% / test {test_acc:.2f}% (L={args.msg_len})")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"decoder_{args.msg_len}bit.pth")
    save_decoder(
        path,
        model,
        b2t_map,
        extra={"test_messages": messages[perm[n_train:]], "seed": args.seed, "test_bit_acc": test_acc},
    )
    print(f"Saved decoder to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark4D Stage 1: X-CLIP guided message decoder")
    parser.add_argument("--msg_len", type=int, required=True, help="L, number of message bits (32 / 48 / 64 in the paper)")
    parser.add_argument("--out_dir", type=str, default="checkpoints/decoder")
    parser.add_argument("--num_messages", type=int, default=5000)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
