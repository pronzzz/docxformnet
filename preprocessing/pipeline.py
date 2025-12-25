import torch
import numpy as np

class Preprocessor:
    def __init__(self, max_seq_length=512, image_size=224):
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        
        # Mean/Std for image normalization (standard ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def normalize_bbox(self, bbox, width, height):
        """
        Normalizes bounding box to 0-1000 scale.
        
        Args:
            bbox: [x0, y0, x1, y1]
            width: image width
            height: image height
        
        Returns:
            normalized_bbox: [x0, y0, x1, y1] checked to be within 0-1000
        """
        x0, y0, x1, y1 = bbox
        
        # Ensure we don't divide by zero
        width = max(width, 1)
        height = max(height, 1)
        
        x0 = int((x0 / width) * 1000)
        y0 = int((y0 / height) * 1000)
        x1 = int((x1 / width) * 1000)
        y1 = int((y1 / height) * 1000)
        
        # Clamp values
        return [
            max(0, min(1000, x0)),
            max(0, min(1000, y0)),
            max(0, min(1000, x1)),
            max(0, min(1000, y1))
        ]

    def process_image(self, image):
        """
        Resizes and normalizes an image.
        
        Args:
            image (PIL.Image): Input image.
            
        Returns:
            torch.Tensor: Preprocessed image tensor (C, H, W).
        """
        image = image.resize((self.image_size, self.image_size))
        img_np = np.array(image).astype(np.float32) / 255.0
        
        if img_np.ndim == 2: # Greyscale
            img_np = np.stack((img_np,)*3, axis=-1)
            
        if img_np.shape[2] == 4: # RGBA
            img_np = img_np[:, :, :3]
            
        # Normalize
        img_np = (img_np - self.mean) / self.std
        
        # Transpose to (C, H, W)
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        return img_tensor

    def construct_adjacency_matrix(self, bboxes, token_count, mode='knn', k=5):
        """
        Constructs a graph adjacency matrix based on spatial relationships.
        
        Args:
            bboxes (list or tensor): [N, 4] normalized bboxes.
            token_count (int): Number of actual tokens found (to mask padding).
            mode (str): 'knn' or 'beta_skeleton'.
            k (int): Number of neighbors for KNN.
            
        Returns:
            torch.Tensor: [N, N] Adjacency matrix.
        """
        # Placeholder for simple KNN based on centers
        if isinstance(bboxes, list):
            bboxes = torch.tensor(bboxes, dtype=torch.float32)

        if bboxes.size(0) == 0:
            return torch.zeros((0, 0))
            
        N = bboxes.size(0)
        adj = torch.zeros((N, N), dtype=torch.float32)
        
        # Calculate centers
        # bbox: x0, y0, x1, y1
        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        centers = torch.stack([centers_x, centers_y], dim=1) # [N, 2]
        
        # Distance matrix
        # dist[i, j] = ||c[i] - c[j]||
        dists = torch.cdist(centers, centers)
        
        if mode == 'knn':
            # Get k nearest neighbors
            # For each node, find indices of top k smallest distances
            # We exclude self-loop usually, or include it. Let's include it as neighbor 0.
            # k+1 because topk returns largest, so we use smallest
            pass
            # topk is for largest, so we use negative dist or sort
            
            _, indices = torch.topk(dists, k=min(k + 1, N), dim=1, largest=False)
            
            for i in range(N):
                if i >= token_count: continue # Ignore padding if we had it here
                
                for neighbor_idx in indices[i]:
                     if neighbor_idx < token_count:
                        adj[i, neighbor_idx] = 1.0
                        
        return adj

if __name__ == "__main__":
    # Test
    prep = Preprocessor()
    bbox = [10, 10, 50, 50]
    norm = prep.normalize_bbox(bbox, 100, 100)
    print(f"Bbox: {bbox} -> Norm: {norm}")
