import argparse
from torch.utils.data import DataLoader
from models import *
from datasets import *
from utils import *
import torch
print(torch.__version__)

parser = argparse.ArgumentParser()

parser.add_argument("-network_name", type=str, default="3cGAN", help="name of the network")
parser.add_argument("--dataset_name", type=str, default="ex-vivo", help="name of the training dataset")
parser.add_argument("--testing_dataset", type=str, default="ex-vivo", help="name of the testing dataset")
parser.add_argument("--lambda_cyc", type=float, default=0.1, help="cycle loss weight")


parser.add_argument("--epoch", type=int, default=50, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=51, help="number of epochs of training")
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
parser.add_argument("--lambda_id", type=float, default=1, help="identity loss weight")
opt = parser.parse_args()
print(opt)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss(size_average=None, reduce=None, reduction = 'elementwise_mean')

cuda = torch.cuda.is_available()
input_shape = (opt.channels, opt.img_height, opt.img_width)
# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)

if cuda:
    G_AB = G_AB.cuda()

G_AB.load_state_dict(torch.load("saved_models/%s-%s-G_AB-%dep.pth" % (opt.network_name, opt.dataset_name, opt.epoch)))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_B1_buffer = ReplayBuffer()
fake_A2_buffer = ReplayBuffer()

transforms_testing_non_fliped_ = [
    transforms.ToTensor(),
]

# Test data loader - non flipped
val_dataloader_non_flipped = DataLoader(
    ImageDataset("../data/Testing/%s-testing" % opt.testing_dataset, transforms_=transforms_testing_non_fliped_, unaligned=False),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

def testing():
    os.makedirs("images/%s-Est-Depths" % (opt.network_name), exist_ok=True)
    G_AB.eval()

    for i, batch in enumerate(val_dataloader_non_flipped):
        real_A = Variable(batch["A"].type(Tensor))
        fake_B1 = G_AB(real_A)
        save_image(fake_B1, "images/%s-Est-Depths/%s-Est-Depths-%s.png" % (opt.network_name,opt.network_name, i),normalize=False, scale_each=False) #range= (0,128)
testing()


