import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pandas as pd
import os
from PIL import Image
import time
from tqdm import tqdm

# ========================
# 1. DATASET CLASS
# ========================
class creepyBikeDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.samples = []
        self.parse_sequence()

    def parse_sequence(self):
        rgb_folder = os.path.join(self.data_dir, 'rgb')
        ground_truth_csv = os.path.join(self.data_dir, 'ground-truth.csv')
        
        ground_truth_dataframe = pd.read_csv(
            ground_truth_csv,
            names=['frame', 'bottom_left_x', 'bottom_left_y', 'top_right_x', 'top_right_y'],
            header=0
        )
        
        for _, row in ground_truth_dataframe.iterrows():
            frame_index = int(row['frame'])
            filename = f"{frame_index}.jpg"
            image_path = os.path.join(rgb_folder, filename)
            
            if not os.path.exists(image_path):
                print(f"Warning: {filename} not found, skipping")
                continue
            
            x_min = row['bottom_left_x']
            y_min = row['top_right_y']
            x_max = row['top_right_x']
            y_max = row['bottom_left_y']
            bounding_box = [x_min, y_min, x_max, y_max]
            self.samples.append((image_path, bounding_box))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, bounding_box = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        x_min, y_min, x_max, y_max = bounding_box
        
        boxes = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = torch.tensor([(x_max - x_min) * (y_max - y_min)], dtype=torch.float32)
        iscrowd = torch.tensor([0], dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, target

def get_transforms(train=False):
    transforms = [T.ToTensor()]
    return T.Compose(transforms)

# ========================
# 2. MODEL SETUP
# ========================
def get_model(num_classes):
    # Load pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# ========================
# 3. TRAINING FUNCTION
# ========================
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        progress_bar.set_postfix({'loss': losses.item()})
    
    return total_loss / len(data_loader)

# ========================
# 4. COLLATE FUNCTION
# ========================
def collate_fn(batch):
    return tuple(zip(*batch))

# ========================
# 5. MAIN TRAINING SCRIPT
# ========================
def main():
    # Configuration
    data_dir = 'data'  # UPDATE THIS PATH to your sequence folder
    num_classes = 2  # 1 class (target) + background
    num_epochs = 10
    batch_size = 2
    learning_rate = 0.005
    
    # Device configuration
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create dataset and data loader
    dataset = creepyBikeDataset(data_dir=data_dir, transforms=get_transforms(train=True))
    print(f"Dataset loaded with {len(dataset)} samples")
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for faster loading
        collate_fn=collate_fn
    )
    
    # Create model
    model = get_model(num_classes)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
        lr_scheduler.step()
        
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), 'faster_rcnn_final.pth')
    print(f"\nTraining complete! Total time: {(time.time() - start_time)/60:.2f} minutes")
    print("Model saved as: faster_rcnn_final.pth")

if __name__ == '__main__':
    main()
