import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

from nightsight import model
from nightsight import data
# TODO Remove hparams dependency - Global var?
from train import hparams


def test(args):
    net = model.FinalNet.load_from_checkpoint(checkpoint_path=weights_path)
    net.eval()

    tf = A.Compose([
        A.Resize(hparams['image_size'],
                 hparams['image_size'],
                 interpolation=4,
                 p=1),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
        ToTensorV2(),
    ])

    ds = ZeroDceDS(args.data_path,
                   args.data_glob,
                   train=False,
                   transform=test_transform)
    dl = D.DataLoader(ds, batch_size=5, pin_memory=False, shuffle=True)
    batch = next(iter(dl))
    plt.imshow(batch[0].permute(1, 2, 0))

    results_1, results, results_A = testing_model(batch)

    display = torch.cat([batch, results]).detach()
    display = torch.clamp(display * torch.Tensor([255]).type_as(results), 0,
                          255)
    if plot:
        display = tv.utils.make_grid(display,
                                     nrow=display.size()[0] // 2,
                                     padding=2,
                                     normalize=False)
        image = display.detach().permute(1, 2, 0).cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(image)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path")
    parser.add_argument("data_path")
    parser.add_argument("data_glob")
    parser.add_argument("plot", type=bool)
    args = parser.parse_args()
    test(args)
