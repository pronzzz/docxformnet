import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

def visualize_document(image_path, words, bboxes, labels=None, output_path=None):
    """
    Visualizes a document with checking bounding boxes and labels.
    
    Args:
        image_path (str): Path to the image file.
        words (list): List of text tokens.
        bboxes (list): List of bounding boxes [x0, y0, x1, y1] (normalized 0-1000).
        labels (list, optional): List of labels for tokens.
        output_path (str, optional): Path to save the visualization.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    fig, ax = plt.subplots(1, figsize=(12, 16))
    ax.imshow(image)
    
    # Generate colors for labels if provided
    label_colors = {}
    
    for i, (word, bbox) in enumerate(zip(words, bboxes)):
        # Normalize bbox (0-1000) to actual image coordinates
        x0, y0, x1, y1 = bbox
        x0 = (x0 / 1000) * width
        y0 = (y0 / 1000) * height
        x1 = (x1 / 1000) * width
        y1 = (y1 / 1000) * height
        
        box_w = x1 - x0
        box_h = y1 - y0
        
        color = 'red'
        label_text = ""
        
        if labels:
            label = labels[i]
            if label not in label_colors:
                label_colors[label] = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            color = label_colors[label]
            label_text = f" [{label}]"

        # Create rectangle patch
        rect = patches.Rectangle((x0, y0), box_w, box_h, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add text
        # ax.text(x0, y0 - 2, word + label_text, fontsize=8, color=color, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
