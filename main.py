import torch
import torch.optim as optim
import numpy as np
import torchvision.utils as vutils
import gym
from tensorboardX import SummaryWriter
import preprocessing as pp
# import matplotlib.pyplot as plt
from tqdm import tqdm
# import cv2
from model import *


log = gym.logger
log.set_level(gym.logger.INFO)

BATCH_SIZE = 16
LATENT_VECTOR_SIZE = 100
DIS_FILTERS = 64
GEN_FILTERS = 64

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000

if __name__ == '__main__':
    device = torch.device("cuda" if (
        torch.cuda.is_available()) else "cpu")
    envs = [pp.InputWrapper(gym.make(env), img_size=IMAGE_SIZE)
            for env in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')]
    input_shape = envs[0].observation_space.sample().shape

    generator = Generator(output_shape=input_shape,
                          filters=GEN_FILTERS,
                          latent_vector_dim=LATENT_VECTOR_SIZE).to(device)
    discriminator = Discriminator(
        input_shape=input_shape, filters=DIS_FILTERS).to(device)

    loss = nn.BCELoss()
    gen_optimizer = optim.Adam(
        params=generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(
        params=discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    true_labels = torch.ones(BATCH_SIZE, dtype=torch.float32, device=device)
    fake_labels = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=device)

    # training details
    dis_losses = []
    gen_losses = []
    writer = SummaryWriter()
    num_iteration = 0
    total_iteration = 50000
    for batch in tqdm(pp.iterate_batches(envs, BATCH_SIZE),
                      total=total_iteration):
        if num_iteration >= total_iteration:
            break
        gen_input = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input = gen_input.normal_(0, 1).to(device)
        batch = batch.to(device)
        gen_output = generator(gen_input)

        # train Discriminator
        dis_optimizer.zero_grad()
        dis_output_true = discriminator(batch)
        # we don't want gradients to gen_output when training the
        # discriminator
        dis_output_fake = discriminator(gen_output.detach())

        dis_loss = loss(dis_output_true, true_labels) + \
            loss(dis_output_fake, fake_labels)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # train Generator
        gen_optimizer.zero_grad()
        dis_output = discriminator(gen_output)
        gen_loss = loss(dis_output, true_labels)
        gen_loss.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss.item())
        num_iteration += 1

        if num_iteration % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e", num_iteration,
                     np.mean(gen_losses), np.mean(dis_losses))
            writer.add_scalar("gen_loss", np.mean(gen_losses), num_iteration)
            writer.add_scalar("dis_loss", np.mean(dis_losses), num_iteration)
            gen_losses = []
            dis_losses = []
        if num_iteration % SAVE_IMAGE_EVERY_ITER == 0:
            # scale the images back to their orginal 0 - 255 range
            for i in range(BATCH_SIZE):
                for j in range(3):
                    gen_output.data[i, :, :, j] = (
                        gen_output.data[i, :, :, j] * 127.5) / 127.5
                    batch.data[i, :, :, j] = (
                        batch.data[i, :, :, j] * 127.5) / 127.5
            writer.add_image("fake", vutils.make_grid(
                gen_output.data[:64]), num_iteration)
            writer.add_image("real", vutils.make_grid(
                batch.data[:64]), num_iteration)
