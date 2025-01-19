import torch
from torch_geometric.data import Data
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
from torch_geometric.nn import GCNConv  # Graph Convolution Network for GNN

# COCO Labels
COCO_LABELS = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Step 1: Load and preprocess the image
def load_and_preprocess_image(image_path, resize_to=(512, 512)):
    image = Image.open(image_path).resize(resize_to)  # Resize to optimize memory usage
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor

# Step 2: Object Detection (Faster R-CNN)
def detect_objects(image_tensor):
    model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions

# Step 3: Create Graph from Object Detections (for GNN)
def create_graph(predictions, boxes, labels):
    num_objects = len(boxes)
    object_names = [COCO_LABELS[label.item()] if label.item() < len(COCO_LABELS) else "Unknown" for label in labels]
    
    # Create edges based on object relationships (e.g., "near" or "on")
    edges = []
    relationships = []  # Initialize the relationships list
    
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            box_i = boxes[i].cpu().numpy()
            box_j = boxes[j].cpu().numpy()
            
            # Add a "near" relationship if objects are within a certain distance
            if np.abs(box_i[0] - box_j[0]) < 100 and np.abs(box_i[1] - box_j[1]) < 100:
                edges.append((i, j))  # "near" relationship
                relationships.append(f"{object_names[i]} near {object_names[j]}")
                
            # Add an "on" relationship if one object is "on" top of another
            if box_i[2] > box_j[0] and box_i[1] < box_j[3]:
                edges.append((i, j))  # "on" relationship
                relationships.append(f"{object_names[i]} on {object_names[j]}")
    
    # Convert object attributes (like box dimensions or class) into node features
    node_features = []
    for box in boxes:
        x1, y1, x2, y2 = box.cpu().numpy()
        width, height = x2 - x1, y2 - y1
        node_features.append([width, height])  # Simple features: width and height
    
    # Convert to PyTorch tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create a PyG Data object
    data = Data(x=node_features, edge_index=edge_index)
    return relationships, object_names, boxes, data

# Step 4: Graph Neural Network Model (GCN)
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Step 5: Training/Inference with the GNN Model
def inference_with_gnn(data):
    model = GNNModel(in_channels=2, hidden_channels=16, out_channels=4)  # Example architecture
    model.eval()
    with torch.no_grad():
        output = model(data)
    return output

# Step 6: Visualize Objects with Bounding Boxes
def visualize_objects_with_bboxes(image, object_names, boxes):
    root = tk.Tk()
    root.title("Image with Object Bounding Boxes")

    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas = tk.Canvas(frame, width=image.width, height=image.height,
                       scrollregion=(0, 0, image.width, image.height))
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    canvas.config(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    h_scrollbar.config(command=canvas.xview)
    v_scrollbar.config(command=canvas.yview)

    image_tk = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

    for i, box in enumerate(boxes):
        box = box.cpu().numpy().astype(int)
        canvas.create_rectangle(box[0], box[1], box[2], box[3], outline="red", width=2)
        canvas.create_text(box[0], box[1], anchor=tk.NW, text=object_names[i], fill="red")

    root.mainloop()

# Step 7: Visualize Relationships with a Graph
def visualize_relationships_graph(relationships, object_names, gnn_output):
    root = tk.Tk()
    root.title("Object Relationship Graph")

    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas = tk.Canvas(frame, width=1200, height=800, scrollregion=(0, 0, 2000, 1500))
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.config(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    h_scrollbar.config(command=canvas.xview)
    v_scrollbar.config(command=canvas.yview)

    node_positions = {}
    spacing_x, spacing_y = 300, 150
    for i, name in enumerate(object_names):
        x, y = 100 + (i % 5) * spacing_x, 100 + (i // 5) * spacing_y
        node_positions[name] = (x, y)
        canvas.create_oval(x-30, y-30, x+30, y+30, fill="yellow")
        canvas.create_text(x, y, text=name, fill="black")

    # Visualize GNN-based relationships
    for relationship in relationships:
        if "near" in relationship:
            obj1, obj2 = relationship.split(" near ")
            rel_type = "near"
        elif "on" in relationship:
            obj1, obj2 = relationship.split(" on ")
            rel_type = "on"
        else:
            continue

        if obj1 in node_positions and obj2 in node_positions:
            x1, y1 = node_positions[obj1]
            x2, y2 = node_positions[obj2]
            canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            canvas.create_text(mid_x, mid_y, text=rel_type, fill="red", font=("Arial", 12))

    root.mainloop()

# Main function
if __name__ == "__main__":
    image_path = "image.jpg"  # Replace with your image path
    image, image_tensor = load_and_preprocess_image(image_path)
    predictions = detect_objects(image_tensor)

    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']

    # Create graph and relationships
    relationships, object_names, boxes, data = create_graph(predictions, boxes, labels)

    # Inference using the GNN model
    gnn_output = inference_with_gnn(data)

    # Now visualize the objects and relationships with GNN output
    visualize_objects_with_bboxes(image, object_names, boxes)
    visualize_relationships_graph(relationships, object_names, gnn_output)
