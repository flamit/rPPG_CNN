import os
import torch
from models.resnet_attention import ResidualNet
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from data_reader import FaceFrameReaderTrain, FaceFrameReaderTest
from tensorboardX import SummaryWriter
from losses import PearsonLoss


parser = ArgumentParser()
parser.add_argument("--image_dir", default="images", type=str, help="Directory where images are located")
parser.add_argument("--image_size", default=256, type=int, help="Face image size")
parser.add_argument("--T", default=64, type=int, help="Number of frames to stack")
parser.add_argument("--N", default=32, type=int, help="Number of grids to divide the image into")
parser.add_argument("--batch_size", default=4, type=int, help="Number of inputs in a batch")
parser.add_argument("--n_threads", default=4, type=int, help="Number of workers for data pipeline")
parser.add_argument("--train", default=True, action='store_true', help="Whether training or evaluating")
parser.add_argument("--epochs", default=1, type=int, help="Number of complete passes over data to train for")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for the optimizer")
parser.add_argument("--save_dir", default='ckpt', type=str, help="Directory for saving trained models")
parser.add_argument("--save_iter", default=5, type=int, help="Save a model ckpt after these iterations")
parser.add_argument("--loss", default='pearson', type=str, help="The loss to use, use either 'l1' or 'pearson'")

def get_data(image_dir):
    image_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if not x.endswith('.txt')]

    return image_paths


def train(model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_names = get_data(args.image_dir)
    data_pipeline = FaceFrameReaderTrain(dir_names, (args.image_size, args.image_size), args.loss, args.T, args.N)
    data_queue = DataLoader(data_pipeline, shuffle=False, batch_size=args.batch_size, num_workers=args.n_threads)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.loss == 'l1':
        loss_function = torch.nn.L1Loss()
    elif args.loss == 'pearson':
        loss_function = PearsonLoss(args.T)
    else:
        print("Chosen loss not recognized")
        raise

    model.to(device)
    model.train()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    writer = SummaryWriter()

    for epoch in range(args.epochs):
        for step, data in enumerate(data_queue):
            opt.zero_grad()
            spatio_tempo, target = data
            logits = model(spatio_tempo.float().to(device))
            loss = loss_function(logits, target.float().to(device))
            if step % args.save_iter == 0:
                print("Epoch: {0}, Step: {1}, Loss: {2}".format(epoch, step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), epoch * len(data_queue) + step)
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, "checkpoint_{0}_{1}.pth".format(epoch, step)))
            loss.backward()
            opt.step()


def predict(model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_names = get_data(args.image_dir)
    data_pipeline = FaceFrameReaderTest(dir_names, (args.image_size, args.image_size), args.T, args.N)
    data_queue = DataLoader(data_pipeline, shuffle=False, batch_size=args.batch_size, num_workers=args.n_threads)

    model.to(device)
    model.eval()

    for data in data_queue:
        preds = model(data.float().to(device))
        predicted_heart_rate = preds.mean()
        print(predicted_heart_rate)


if __name__ == "__main__":
    args = parser.parse_args()
    resnet18 = ResidualNet("ImageNet", 18, args.T, 'BAM', args.loss)
    if args.train:
        train(resnet18, args)
    else:
        predict(resnet18, args)
