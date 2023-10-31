from timm.data.transforms import RandomResizedCropAndInterpolation
from PIL import Image
from matplotlib import pyplot as plt

tfm = RandomResizedCropAndInterpolation(size=384, scale=(0.8, 1))
X   = Image.open("/public/share/others/challenge_dataset/MosquitoAlert2023/phase2/cls_images_png/aegypti/sub_train_00974.png")
plt.imshow(X)

for i in range(32):
    y = tfm(X)
    y.save(f'test_imgs/test_{i}.png')