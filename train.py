import sys
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import interpolate

from model.build_model import build_netG, build_netD
from data.customdataset import CustomDataset
from model.losses import gdloss
from utils.util import new_state_dict
from options import Options

from torch.utils.data import DataLoader

import torch.nn.functional as F

if __name__ == "__main__":
    use_gpu = False
    opt = Options().parse()
    opt.phase = "train"
    print(opt)

    data_set = CustomDataset(opt)
    print("Image numbers:", data_set.img_size)
    # print("Size of data:", data_set.__getitem__(index=0))

    dataloader = torch.utils.data.DataLoader(
        data_set, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers)
    )

    for i, data in enumerate(dataloader):
        print("DATA: ", data)
        high_real_patches = data[
            "high_img_patches"
        ]  # Extracting high-resolution patches from data
        for k in range(0, opt.num_patches):
            high_real_patch = high_real_patches[:, k]  # Access individual patches
            print(
                "high_real_patch size:", high_real_patch.shape
            )  # Now this should work as it's inside the loop

            # Ensure low_patch is interpolated back to the exact size of high_real_patch
            low_patch = interpolate(high_real_patch, scale_factor=0.5)

    generator = build_netG(opt)
    discriminator, target_real, target_fake = build_netD(opt)

    if opt.gpu_ids != "-1":
        num_gpus = len(opt.gpu_ids.split(","))
    else:
        num_gpus = 0
    print("number of GPU:", num_gpus)

    if opt.resume:
        generator.load_state_dict(new_state_dict(opt.generatorWeights))
        discriminator.load_state_dict(new_state_dict(opt.discriminatorWeights))
        print("Weight is loaded")
    else:
        pretrainW = "./checkpoints/g_pre-train.pth"
        if os.path.exists(pretrainW):
            generator.load_state_dict(new_state_dict(pretrainW))
            print("Pre-Trained G Weight is loaded")

    adversarial_criterion = nn.MSELoss()  # nn.BCELoss()

    if (opt.gpu_ids != -1) & torch.cuda.is_available():
        use_gpu = True
        generator.cuda()
        discriminator.cuda()
        adversarial_criterion.cuda()
        target_real = target_real.cuda()
        target_fake = target_fake.cuda()
        if num_gpus > 1:
            generator = nn.DataParallel(generator)
            discriminator = nn.DataParallel(discriminator)

    optim_generator = optim.Adam(
        generator.parameters(), lr=opt.generatorLR, weight_decay=1e-4
    )
    optim_discriminator = optim.Adam(
        discriminator.parameters(), lr=opt.discriminatorLR, weight_decay=1e-4
    )
    StepLR_G = torch.optim.lr_scheduler.StepLR(
        optim_generator, step_size=10, gamma=0.85
    )
    StepLR_D = torch.optim.lr_scheduler.StepLR(
        optim_discriminator, step_size=10, gamma=0.85
    )

    print("start training")

    for epoch in range(opt.nEpochs):
        mean_generator_adversarial_loss = 0.0
        mean_generator_l2_loss = 0.0
        mean_generator_gdl_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_discriminator_loss = 0.0

        for i, data in enumerate(dataloader):
            # get input data
            high_real_patches = data[
                "high_img_patches"
            ]  # [batch_size,num_patches,C,D,H,W]

            print("high_real_patch size:", high_real_patch.shape)

            for k in range(0, opt.num_patches):
                high_real_patch = high_real_patches[:, k]  # [BCDHW]
                print(
                    "high_real_patch size before interpolation:", high_real_patch.shape
                )

                # Downscale by half
                low_patch = interpolate(high_real_patch, scale_factor=0.5)
                print("low_patch size after downscale:", low_patch.shape)

                # Generate high-resolution output from the low-resolution input
                high_gen = generator(low_patch)

                # Upscale to match high_real_patch dimensions
                high_gen = interpolate(
                    high_gen, size=high_real_patch.shape[2:]
                )  # Ensure exact dimension match
                print(
                    "high_gen size after upscale to match high_real_patch:",
                    high_gen.shape,
                )

                print("Shape of high_real_patch:", high_real_patch.shape)
                print("Shape of high_gen:", high_gen.shape)

                if use_gpu:
                    high_real_patch = high_real_patch.cuda()
                    # generate fake data
                    high_gen = generator(low_patch.cuda())
                else:
                    high_gen = generator(low_patch)

                print("high_gen size after generator:", high_gen.shape)

                ######### Train discriminator #########
                discriminator.zero_grad()

                real_output = discriminator(high_real_patch)
                fake_output = discriminator(high_gen.detach())

                # Print outputs' shapes to verify dimensions
                print("Shape of real output from discriminator:", real_output.shape)
                print("Shape of fake output from discriminator:", fake_output.shape)

                # Print the shapes to verify
                print("Shape of real output:", real_output.shape)
                print("Shape of fake output:", fake_output.shape)
                print("Shape of target real:", target_real.shape)
                print("Shape of target fake:", target_fake.shape)

                print("Shape of high_real_patch:", high_real_patch.shape)
                print("Shape of high_gen:", high_gen.shape)

                # Dynamically create targets based on discriminator output
                target_real = torch.ones_like(real_output)
                target_fake = torch.zeros_like(fake_output)

                discriminator_loss_real = 0.5 * adversarial_criterion(
                    real_output, target_real
                )
                discriminator_loss_fake = 0.5 * adversarial_criterion(
                    fake_output, target_fake
                )
                discriminator_loss = discriminator_loss_real + discriminator_loss_fake

                # Print losses to monitor training progression
                print(
                    "Discriminator loss for real data:", discriminator_loss_real.item()
                )
                print(
                    "Discriminator loss for fake data:", discriminator_loss_fake.item()
                )

                mean_discriminator_loss += (
                    discriminator_loss.item()
                )  # Update running total
                discriminator_loss.backward()
                optim_discriminator.step()

                # Overall discriminator loss for current batch
                print(
                    "Total discriminator loss for current batch:",
                    discriminator_loss.item(),
                )

                ######### Train generator #########
                generator.zero_grad()

                # Before using high_gen for calculating gdloss
                if high_gen.size() != high_real_patch.size():
                    # Adjust high_gen to match the size of high_real_patch
                    high_gen = F.interpolate(
                        high_gen,
                        size=(
                            high_real_patch.size(2),
                            high_real_patch.size(3),
                            high_real_patch.size(4),
                        ),
                        mode="trilinear",
                        align_corners=False,
                    )

                # Generator losses calculation
                generator_gdl_loss = opt.gdl * gdloss(high_real_patch, high_gen)
                mean_generator_gdl_loss += (
                    generator_gdl_loss.item()
                )  # Update running total

                generator_l2_loss = nn.MSELoss()(high_real_patch, high_gen)
                mean_generator_l2_loss += (
                    generator_l2_loss.item()
                )  # Update running total

                discriminator_output = discriminator(
                    high_gen
                )  # Discriminator output for generated images
                generator_adversarial_loss = adversarial_criterion(
                    discriminator_output, target_real
                )
                mean_generator_adversarial_loss += (
                    generator_adversarial_loss.item()
                )  # Update running total

                # Total generator loss
                generator_total_loss = (
                    generator_gdl_loss
                    + generator_l2_loss
                    + opt.advW * generator_adversarial_loss
                )
                mean_generator_total_loss += (
                    generator_total_loss.item()
                )  # Update running total

                # Backpropagation
                generator_total_loss.backward()
                optim_generator.step()

                # Print loss values to monitor training
                print(f"Generator GDL Loss: {generator_gdl_loss.item():.4f}")
                print(f"Generator L2 Loss: {generator_l2_loss.item():.4f}")
                print(
                    f"Generator Adversarial Loss: {generator_adversarial_loss.item():.4f}"
                )
                print(f"Total Generator Loss: {generator_total_loss.item():.4f}")

                # Optional: print shapes of the patches and generated images for debugging
                print("Shape of real patches:", high_real_patch.shape)
                print("Shape of generated patches:", high_gen.shape)

            ######### Status and display #########
            sys.stdout.write(
                "\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (GDL/L2/Adv/Total): %.4f/%.4f/%.4f/%.4f"
                % (
                    epoch,
                    opt.nEpochs,
                    i,
                    len(dataloader),
                    discriminator_loss,
                    generator_gdl_loss,
                    generator_l2_loss,
                    generator_adversarial_loss,
                    generator_total_loss,
                )
            )

        StepLR_G.step()
        StepLR_D.step()

        # Before saving checkpoints, ensure the directory exists
        if not os.path.exists(opt.checkpoints_dir):
            os.makedirs(
                opt.checkpoints_dir
            )  # Create the directory if it does not exist

        if epoch % opt.save_fre == 0:
            # Do checkpointing
            torch.save(generator.state_dict(), "%s/g.pth" % opt.checkpoints_dir)
            torch.save(discriminator.state_dict(), "%s/d.pth" % opt.checkpoints_dir)

        sys.stdout.write(
            "\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (GDL/L2/Adv/Total): %.4f/%.4f/%.4f/%.4f\n"
            % (
                epoch,
                opt.nEpochs,
                i,
                len(dataloader),
                mean_discriminator_loss / len(dataloader) / opt.num_patches,
                mean_generator_gdl_loss / len(dataloader) / opt.num_patches,
                mean_generator_l2_loss / len(dataloader) / opt.num_patches,
                mean_generator_adversarial_loss / len(dataloader) / opt.num_patches,
                mean_generator_total_loss / len(dataloader) / opt.num_patches,
            )
        )

    for i, data in enumerate(dataloader):
        print(data)
