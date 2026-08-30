import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

img_top = mpimg.imread('visualizations_GDELT/combined_difficulty.png')
img_bottom = mpimg.imread('visualizations_ICEWS14/combined_difficulty.png')

fig = plt.figure(figsize=(7.5, 7.0))
gs = GridSpec(2, 1, figure=fig, hspace=0, wspace=0,
              left=0, right=1, top=1, bottom=0)  # zero all margins

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

ax1.imshow(img_top, aspect='auto')
ax1.axis('off')

ax2.imshow(img_bottom, aspect='auto')
ax2.axis('off')

plt.savefig('combined_two_datasets.tiff', dpi=600)