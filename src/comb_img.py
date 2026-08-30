import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_top = mpimg.imread('visualizations_GDELT/combined_difficulty.png')
img_bottom = mpimg.imread('visualizations_ICEWS14/combined_difficulty.png')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.0))

ax1.imshow(img_top)
ax1.axis('off')

ax2.imshow(img_bottom)
ax2.axis('off')

plt.subplots_adjust(hspace=0, wspace=0)  # no gap between the two
plt.savefig('combined_two_datasets.tiff', dpi=600, bbox_inches='tight', pad_inches=0)