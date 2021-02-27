import rawpy
import imageio
import neptune
from neptune import Session
from getpass import getpass

# Get Neptune API token
api_token = getpass("Enter Neptune.ai API token: ")

# Initialize Neptune project
session = Session.with_default_backend(api_token=api_token)
project = session.get_project('rshwndsz/nightsight')
experiment = project.get_experiments(id='NIG-11')[0]
experiment

# Download checkpoint from Neptune
artifact_path   = 'epoch=133-avg_val_loss=1.06.ckpt'
artifact_name   = artifact_path.split('/')[-1]
checkpoint_dir  = os.path.join('checkpoints', 'downloads')
checkpoint_path = os.path.join(checkpoint_dir, artifact_name)

experiment.download_artifact(path=artifact_path, destination_dir=checkpoint_dir)
testing_model = FinalNet.load_from_checkpoint(checkpoint_path=checkpoint_path)

testing_model.eval()

### MIT - Adobe 5K

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# pip install rawpy
# mkdir -p data/test_data/Adobe-5k && cd data/test_data/Adobe-5k
# wget 'https://data.csail.mit.edu/graphics/fivek/img/dng/a0005-jn_2007_05_10__564.dng'
# wget 'https://data.csail.mit.edu/graphics/fivek/img/dng/a0010-jmac_MG_4807.dng'
# wget 'https://data.csail.mit.edu/graphics/fivek/img/dng/a0018-kme_234.dng'
# wget 'https://data.csail.mit.edu/graphics/fivek/img/dng/a0089-jn_20080509_245.dng'
# wget 'https://data.csail.mit.edu/graphics/fivek/img/dng/a0163-WP_CRW_0638.dng'
# wget 'https://data.csail.mit.edu/graphics/fivek/img/dng/a0623-dvf_031.dng'
# cd ../../../

urls = [
         'https://data.csail.mit.edu/graphics/fivek/img/dng/a0005-jn_2007_05_10__564.dng',
         'https://data.csail.mit.edu/graphics/fivek/img/dng/a0010-jmac_MG_4807.dng',
         'https://data.csail.mit.edu/graphics/fivek/img/dng/a0018-kme_234.dng',
         'https://data.csail.mit.edu/graphics/fivek/img/dng/a0089-jn_20080509_245.dng',
         'https://data.csail.mit.edu/graphics/fivek/img/dng/a0163-WP_CRW_0638.dng',
]
files = [url.split('/')[-1] for url in urls]

root = os.path.join('data', 'test_data', 'Adobe-5k')

for f in files:
    path = os.path.join(root, f)
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess()
        image = Image.fromarray(rgb)
        image.save(path[:-4] + '.jpg')

test_transform = A.Compose(
    [
        A.Resize(hparams['image_size'], hparams['image_size'], interpolation=4, p=1),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
        ToTensorV2(),
    ]
)
ds = ZeroDceDS("data/test_data/Adobe-5k", "*.jpg", train=False, transform=test_transform)
dl = D.DataLoader(ds, batch_size=5, pin_memory=False, shuffle=True)
batch = next(iter(dl))
plt.imshow(batch[0].permute(1, 2, 0))

# Commented out IPython magic to ensure Python compatibility.
# %%time
# results_1, results, results_A = testing_model(batch)

display = torch.cat([batch, results]).detach()
# display = torch.clamp(display * torch.Tensor([255]).type_as(results), 0, 255)
display = tv.utils.make_grid(display, nrow=display.size()[0]//2, padding=2, normalize=False)
image   = display.detach().permute(1, 2, 0).cpu().numpy()
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.imshow(image)
plt.show()

experiment.log_image("adobe-5k-Nov1", fig)

