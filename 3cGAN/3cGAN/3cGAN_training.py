if __name__ == '__main__':

    import argparse
    import itertools
    from torch.utils.data import DataLoader
    from models import *
    from datasets import *
    from utils import *
    import torch

    parser = argparse.ArgumentParser(description="3cGAN")
    parser.add_argument("-network_name", type=str, default="3cGAN", help="name of the network")
    parser.add_argument("--training_dataset", type=str, default="ex-vivo", help="name of the dataset")
    parser.add_argument("--testing_dataset", type=str, default="ex-vivo", help="name of the testing dataset")
    parser.add_argument("--lambda_merging", type=float, default=10, help="scaling factor for the new loss")
    parser.add_argument("--lambda_cyc", type=float, default=1, help="cycle loss weight")

    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=51, help="number of epochs oef training")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=25, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=200, help="size of image height")
    parser.add_argument("--img_width", type=int, default=200, help="size of image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
    parser.add_argument("--textfile_training_results_interval", type=int, default=50, help="textfile_training_results_interval")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    opt = parser.parse_args()
    print(opt)

    # Create sample and checkpoint directories
    os.makedirs("saved_models/%s-%s" % (opt.network_name, opt.training_dataset), exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.MSELoss()
    criterion_identity = torch.nn.L1Loss()

    cuda = torch.cuda.is_available()
    input_shape = (opt.channels, opt.img_height, opt.img_width)
    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_CB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BC = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_AC = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_CA = GeneratorResNet(input_shape, opt.n_residual_blocks)
    D_B1 = Discriminator(input_shape)
    D_A2 = Discriminator(input_shape)
    D_B3 = Discriminator(input_shape)
    D_C4 = Discriminator(input_shape)
    D_C5 = Discriminator(input_shape)
    D_A6 = Discriminator(input_shape)

    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        G_CB = G_CB.cuda()
        G_BC = G_BC.cuda()
        G_AC = G_AC.cuda()
        G_CA = G_CA.cuda()

        D_B1 = D_B1.cuda()
        D_A2 = D_A2.cuda()
        D_B3 = D_B3.cuda()
        D_C4 = D_C4.cuda()
        D_C5 = D_C5.cuda()
        D_A6 = D_A6.cuda()

        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load("saved_models/%s-%s/G_AB_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        G_BA.load_state_dict(torch.load("saved_models/%s-%s/G_BA_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        G_CB.load_state_dict(torch.load("saved_models/%s-%s/G_CB_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        G_BC.load_state_dict(torch.load("saved_models/%s-%s/G_BC_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        G_AC.load_state_dict(torch.load("saved_models/%s-%s/G_AC_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        G_CA.load_state_dict(torch.load("saved_models/%s-%s/G_CA_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))

        D_B1.load_state_dict(torch.load("saved_models/%s-%s/D_B1_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        D_A2.load_state_dict(torch.load("saved_models/%s-%s/D_A2_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        D_B3.load_state_dict(torch.load("saved_models/%s-%s/D_B3_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        D_C4.load_state_dict(torch.load("saved_models/%s-%s/D_C4_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        D_C5.load_state_dict(torch.load("saved_models/%s-%s/D_C5_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        D_A6.load_state_dict(torch.load("saved_models/%s-%s/D_A6_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))

    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        G_CB.apply(weights_init_normal)
        G_BC.apply(weights_init_normal)
        G_AC.apply(weights_init_normal)
        G_CA.apply(weights_init_normal)

        D_B1.apply(weights_init_normal)
        D_A2.apply(weights_init_normal)
        D_B3.apply(weights_init_normal)
        D_C4.apply(weights_init_normal)
        D_C5.apply(weights_init_normal)
        D_A6.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters(), G_CB.parameters(), G_BC.parameters(), G_AC.parameters(), G_CA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_B1 = torch.optim.Adam(D_B1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_A2 = torch.optim.Adam(D_A2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B3 = torch.optim.Adam(D_B3.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_C4 = torch.optim.Adam(D_C4.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_C5 = torch.optim.Adam(D_C5.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_A6 = torch.optim.Adam(D_A6.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_B1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_A2 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_B3 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B3, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_C4 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_C4, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_C5 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_C5, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_A6 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A6, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_B1_buffer = ReplayBuffer()
    fake_A2_buffer = ReplayBuffer()
    fake_B3_buffer = ReplayBuffer()
    fake_C4_buffer = ReplayBuffer()
    fake_C5_buffer = ReplayBuffer()
    fake_A6_buffer = ReplayBuffer()


    # Image transformations
    transforms_ = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]


    # Training data loader
    dataloader = DataLoader(
        ImageDataset("../data/Training/%s-training" % opt.training_dataset, transforms_=transforms_, unaligned=True),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            real_C = Variable(batch["C"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A2.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A2.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()
            G_CB.train()
            G_BC.train()
            G_AC.train()
            G_CA.train()

            optimizer_G.zero_grad()


            # GAN loss
            fake_B1 = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B1(fake_B1), valid)
            fake_A2 = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A2(fake_A2), valid)

            fake_B3 = G_CB(real_C)
            loss_GAN_CB = criterion_GAN(D_B3(fake_B3), valid)
            fake_C4 = G_BC(real_B)
            loss_GAN_BC = criterion_GAN(D_C4(fake_C4), valid)

            fake_C5 = G_AC(real_A)
            loss_GAN_AC = criterion_GAN(D_C5(fake_C5), valid)
            fake_A6 = G_CA(real_C)
            loss_GAN_CA = criterion_GAN(D_A6(fake_A6), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA + loss_GAN_BC + loss_GAN_CB + loss_GAN_AC + loss_GAN_BC) / 6

            # Cycle loss
            recov_BA = G_BA(fake_B1)
            loss_cycle_BA = criterion_cycle(recov_BA, real_A)
            recov_AB = G_AB(fake_A2)
            loss_cycle_AB = criterion_cycle(recov_AB, real_B)

            recov_BC = G_BC(fake_B3)
            loss_cycle_BC = criterion_cycle(recov_BC, real_C)
            recov_CB = G_CB(fake_C4)
            loss_cycle_CB = criterion_cycle(recov_CB, real_B)

            recov_AC = G_AC(fake_A6)
            loss_cycle_AC = criterion_cycle(recov_AC, real_C)
            recov_CA = G_CA(fake_C5)
            loss_cycle_CA = criterion_cycle(recov_CA, real_A)


            # merging loss:
            recov_253461 = G_AB(G_CA(G_BC(G_CB(G_AC(G_BA(real_B))))))
            loss_cycle_253461 = criterion_cycle(recov_253461, real_B)


            loss_cycle = (loss_cycle_BA + loss_cycle_AB + loss_cycle_BC + loss_cycle_CB + loss_cycle_CA + loss_cycle_AC) / 6 + opt.lambda_merging*loss_cycle_253461

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A2
            # -----------------------

            optimizer_D_A2.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A2(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A2_ = fake_A2_buffer.push_and_pop(fake_A2)
            loss_fake = criterion_GAN(D_A2(fake_A2_.detach()), fake)
            # Total loss
            loss_D_A2 = (loss_real + loss_fake) / 2

            loss_D_A2.backward()
            optimizer_D_A2.step()

            # -----------------------
            #  Train Discriminator B1
            # -----------------------

            optimizer_D_B1.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B1(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B1_ = fake_B1_buffer.push_and_pop(fake_B1)
            loss_fake = criterion_GAN(D_B1(fake_B1_.detach()), fake)
            # Total loss
            loss_D_B1 = (loss_real + loss_fake) / 2

            loss_D_B1.backward()
            optimizer_D_B1.step()

            # -----------------------
            #  Train Discriminator B3
            # -----------------------

            optimizer_D_B3.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B3(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B3_ = fake_B3_buffer.push_and_pop(fake_B3)
            loss_fake = criterion_GAN(D_B3(fake_B3_.detach()), fake)
            # Total loss
            loss_D_B3 = (loss_real + loss_fake) / 2

            loss_D_B3.backward()
            optimizer_D_B3.step()

            # -----------------------
            #  Train Discriminator C4
            # -----------------------

            optimizer_D_C4.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_C4(real_C), valid)
            # Fake loss (on batch of previously generated samples)
            fake_C4_ = fake_C4_buffer.push_and_pop(fake_C4)
            loss_fake = criterion_GAN(D_C4(fake_C4_.detach()), fake)
            # Total loss
            loss_D_C4 = (loss_real + loss_fake) / 2

            loss_D_C4.backward()
            optimizer_D_C4.step()


            # -----------------------
            #  Train Discriminator C5
            # -----------------------

            optimizer_D_C5.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_C5(real_C), valid)
            # Fake loss (on batch of previously generated samples)
            fake_C5_ = fake_C5_buffer.push_and_pop(fake_C5)
            loss_fake = criterion_GAN(D_C5(fake_C5_.detach()), fake)
            # Total loss
            loss_D_C5 = (loss_real + loss_fake) / 2

            loss_D_C5.backward()
            optimizer_D_C5.step()

            # -----------------------
            #  Train Discriminator A6
            # -----------------------

            optimizer_D_A6.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A6(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A6_ = fake_A6_buffer.push_and_pop(fake_A6)
            loss_fake = criterion_GAN(D_A6(fake_A6_.detach()), fake)
            # Total loss
            loss_D_A6 = (loss_real + loss_fake) / 2

            loss_D_A6.backward()
            optimizer_D_A6.step()

            loss_D = (loss_D_A2 + loss_D_B1 + loss_D_B3 + loss_D_C4 + loss_D_C5 + loss_D_A6) / 6

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    opt.lambda_cyc*loss_cycle.item(),
                    #loss_identity.item(),
                    time_left,
                )
            )



        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A2.step()
        lr_scheduler_D_B1.step()
        lr_scheduler_D_B3.step()
        lr_scheduler_D_C4.step()
        lr_scheduler_D_C5.step()
        lr_scheduler_D_A6.step()



        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/%s-%s/%s-%s-G_AB-%dep.pth" % (opt.network_name,opt.training_dataset, opt.network_name,opt.training_dataset, epoch))
            torch.save(G_BA.state_dict(), "saved_models/%s-%s/%s-%s-G_BA-%dep.pth" % (opt.network_name,opt.training_dataset, opt.network_name,opt.training_dataset, epoch))
            torch.save(G_CB.state_dict(), "saved_models/%s-%s/%s-%s-G_CB-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
            torch.save(G_BC.state_dict(), "saved_models/%s-%s/%s-%s-G_BC-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
            torch.save(G_AC.state_dict(), "saved_models/%s-%s/%s-%s-G_AC-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
            torch.save(G_CA.state_dict(), "saved_models/%s-%s/%s-%s-G_CA-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))

            torch.save(D_B1.state_dict(), "saved_models/%s-%s/%s-%s-D_B1-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name,opt.training_dataset, epoch))
            torch.save(D_A2.state_dict(), "saved_models/%s-%s/%s-%s-D_A2-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name,opt.training_dataset, epoch))
            torch.save(D_B3.state_dict(), "saved_models/%s-%s/%s-%s-D_B3-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
            torch.save(D_C4.state_dict(), "saved_models/%s-%s/%s-%s-D_C4-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
            torch.save(D_C5.state_dict(), "saved_models/%s-%s/%s-%s-D_C5-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
            torch.save(D_A6.state_dict(), "saved_models/%s-%s/%s-%s-D_A6-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))





