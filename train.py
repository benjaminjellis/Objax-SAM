"""
Experiment using SAM + SGD + Momentum with CIFAR10 and OneCycleLR
"""
import argparse
import objax
from objax.zoo.wide_resnet import WideResNet
from SAM import SAM
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Normalize
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default = "SAM", type = str, help= "Training mode, either 'SAM  or 'SGD'")
    parser.add_argument("--epochs", default = 100, type = int, help= "Number of epochs to train for (int)")
    parser.add_argument("--learning_rate", default = 0.016, type=float, help= "Starting learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help= "SGD Momentum")
    parser.add_argument("--rho", default=0.05, type=int, help= "Rho parameter for SAM")
    args = parser.parse_args()

    normalisation = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        normalisation
    ])

    test_transform = Compose([
        ToTensor(),
        normalisation
    ])

    train_ds = CIFAR10(root = "./data", train = True, transform = train_transform, download = True)
    test_ds = CIFAR10(root = "./data", train = False, transform = train_transform, download = True)

    train_dl = DataLoader(dataset = train_ds, batch_size = 120, shuffle = False, num_workers = 2)
    test_dl = DataLoader(dataset = test_ds, batch_size = 120, shuffle = False, num_workers = 2)

    model = WideResNet(nin=3, nclass=10, depth=28, width=2)

    @objax.Function.with_vars(model.vars())
    def loss(x, label):
        logit = model(x, training = True)
        return objax.functional.loss.cross_entropy_logits_sparse(logit, label).mean()

    gv = objax.GradValues(loss, model.vars())

    if args.mode == "SAM":
        print("Using SAM")
        opt = SAM(model.vars(), base_optimizer = objax.optimizer.Momentum, momentum = args.momentum)

        @objax.Function.with_vars(model.vars() + gv.vars() + opt.vars())
        def train_op(x, y, lr):
            g, v = gv(x, y)
            opt.first_step(rho = args.rho, grads = g)
            g, v = gv(x, y)
            opt.second_step(lr = lr, grads = g)
            return v

    elif args.mode == "SGD":
        opt = objax.optimizer.Momentum(model.vars(), momentum = args.momentum)

        @objax.Function.with_vars(model.vars() + gv.vars() + opt.vars())
        def train_op(x, y, lr):
            g, v = gv(x, y)
            opt(lr = lr, grads = g)
            return v

    else:
        raise ValueError("Got unexpected mode, expected either 'SAM' or 'SGD'")


    @objax.Function.with_vars(model.vars())
    def eval_batch(images, targets):
        predictions = model(images, training = False)
        correct_this_batch = (predictions.argmax(1) == targets) .sum()
        return correct_this_batch

    eval_batch = objax.Jit(eval_batch)
    train_op = objax.Jit(train_op)

    for epoch in range(args.epochs):
        train_dl = tqdm(train_dl)
        loss = 0
        for i, (images, targets) in enumerate(train_dl):
            print(train_op(images.numpy(), targets.numpy(), args.learning_rate))
            loss += 0.1

        num_correct = 0
        total = test_ds.__len__()
        test_dl = tqdm(test_dl)
        for i, (images, targets) in enumerate(test_dl):
            num_correct += eval_batch(images.numpy(), targets.numpy())

        print(f"Epoch: {epoch + 1} | Accuracy : {num_correct / total * 100} | Train Loss: {loss}")
