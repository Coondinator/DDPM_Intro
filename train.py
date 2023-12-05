import wandb
import torch
import torchvision
from torchvision import datasets
import numpy as np

from diffusion.argument import Arguments
from diffusion.diffusion import GaussianModel, generate_linear_schedule
from diffusion.denoiser import UNet
from torch.utils.data import dataset, Dataset, DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)


def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])

dataset = datasets.CIFAR10(
    root='./cifar_test',
    train=False,
    download=True,
    transform=get_transform(),
)

train_size = int(0.7*len(dataset))
test_size = len(dataset)-train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
test_dataloader = DataLoader(dataset=test_data, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
print(len(test_dataloader))

learning_rate = 0.0025
iterations = 2000


def get_model():
    config_name = 'config.yaml'

    betas = generate_linear_schedule(T=1000, low=1e-4, high=0.02)

    denoiser = UNet(img_channels=3,
                    base_channels=128,
                    channel_mults=(1, 2, 2, 2),
                    num_res_blocks=2,
                    time_emb_dim=128 * 4,
                    norm="gn",
                    dropout=0.1,
                    attention_resolutions=(1,),)
    diffusion = GaussianModel(model=denoiser,
                              img_size=(32, 32),
                              img_channel=3,
                              loss_type='l2',
                              betas=betas).to(device)

    return diffusion

def basic_train(model, save_path, use_labels=False):

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    wandb.init(
        # set the wandb project where this run will be logged
        project="single--test",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.0001,
            "architecture": "DDPM",
            "dataset": "CIFAR10",
            "iteration": 2000,
        }
    )



    for iteration in range(1, iterations + 1):
        acc_train_loss = 0
        test_loss = 0
        print("Start training: iteration ", iteration)
        model.train()

        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            if  use_labels:
                loss = model(x, y)
            else:
                loss = model(x)

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        test_loss = 0
        with torch.no_grad():
            model.eval()
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)

                if  use_labels:
                    loss = model(x, y)
                else:
                    loss = model(x)

                test_loss += loss.item()

            test_loss /= len(test_dataloader)
            acc_train_loss /= len(train_dataloader)

        print('[{:03d}/{}] acc_train_loss: {:.4f}\t test_loss: {:.4f}'.format(
                iteration, iterations, acc_train_loss, test_loss))

        wandb.log({
                "test_loss": test_loss,
                "train_loss": acc_train_loss,
        })



        if iteration % 500 == 0:
            model_filename = f"DDPM-iteration-{iteration}-model.pth"
            optim_filename = f"DDPM-iteration-{iteration}-optim.pth"

            torch.save(model.state_dict(), model_filename)
            torch.save(optimizer.state_dict(), optim_filename)

    wandb.finish()


if __name__ == '__main__':

    save_path = './Model.pt'
    model = get_model()
    basic_train(model, save_path)

