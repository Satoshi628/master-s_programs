import torch
from dalle_pytorch import DiscreteVAE, DALLE

vae = DiscreteVAE(
    image_size = 256,
    num_layers = 3,
    num_tokens = 8192,
    codebook_dim = 1024,
    hidden_dim = 64,
    num_resnet_blocks = 1,
    temperature = 0.9
)

dalle = DALLE(
    dim = 1024,
    vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens = 10000,    # vocab size for text
    text_seq_len = 256,         # text sequence length
    depth = 12,                 # should aim to be 64
    heads = 16,                 # attention heads
    dim_head = 64,              # attention head dimension
    attn_dropout = 0.1,         # attention dropout
    ff_dropout = 0.1            # feedforward dropout
)

dalle = dalle.cuda()
text = torch.randint(0, 10000, (4, 256)).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

# loss = dalle(text, images, return_loss = True)
# loss.backward()

# do the above for a long time with a lot of data ... then

images = dalle.generate_images(text)
images.shape # (4, 3, 256, 256)