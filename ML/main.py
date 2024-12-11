import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
from data_reader import get_data_loader
from incremental_model import MLPModel as IncrementalMLPModel, train_incremental, predict_incremental
from batch_model import Batch_MLPModel, train_batch_model, predict_batch_model
from distance_calculator import DistanceCalculator
from adaptive_window import AdaptiveWindow
from scipy.stats import mode

def gaussian_kernel(distance, sigma=20.0):
    return np.exp(-distance**2 / (2 * sigma**2))

def kmeans_torch(X, num_clusters, iterations=10):
    N, D = X.shape
    centroids = X[torch.randperm(N)[:num_clusters]]  # randomly initialize centroids
    for _ in range(iterations):
        dists = torch.cdist(X, centroids)  # compute distances to centroids
        labels = torch.argmin(dists, dim=1)  # assign labels
        centroids = torch.stack([X[labels == k].mean(dim=0) for k in range(num_clusters)])  # recompute centroids
    return labels, centroids

class ModelHistory:
    def __init__(self):
        self.history = {}

    def add_model(self, model_state, data_features):
        self.history[tuple(data_features.flatten())] = model_state

    def find_closest_model(self, current_features):
        min_distance = float('inf')
        closest_state = None
        for features, state in self.history.items():
            distance = np.linalg.norm(np.array(features) - current_features.flatten())
            if distance < min_distance:
                min_distance = distance
                closest_state = state
        return closest_state

def main():
    parser = argparse.ArgumentParser(description='Run the machine learning model on specified dataset.')
    parser.add_argument('filepath', type=str, help='Path to your data file.')
    args = parser.parse_args()

    batch_size = 1024
    num_features = 10
    num_classes = 2

    data_loader = get_data_loader(args.filepath, batch_size)
    incremental_model = IncrementalMLPModel(num_features, num_classes)
    batch_model = Batch_MLPModel(num_features, num_classes)
    dist_calc = DistanceCalculator(num_features)
    model_history = ModelHistory()
    adaptive_window = AdaptiveWindow(window_size=5, max_batches=5)

    # Optimizers and loss function
    optimizer_incremental = optim.Adam(incremental_model.parameters(), lr=0.001)
    optimizer_batch = optim.Adam(batch_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    last_labeled_data = None
    last_labels = None
    accuracies = []
    last_batch_data = None

    for batch_data in data_loader:
        features, labels = batch_data[:, :-1], batch_data[:, -1].long()  # 确保标签是 long 类型
        #print("Batch processed with features shape:", features.shape)  # 打印特征形状
        adaptive_window.add_batch(batch_data.numpy())
        #print("New batch added to adaptive window")
        if last_batch_data is not None:
            shift_distance = dist_calc.calculate_shift_distance(features.numpy(), last_batch_data.numpy())
            shift_type = dist_calc.classify_shift(shift_distance)
            print("Shift type:", shift_type)  # 打印位移类型
            if shift_type == "Severe shift":
                closest_state = model_history.find_closest_model(features.numpy())
                #print("Major")
                print("Closest model state found:", closest_state is not None)  # 检查是否找到最接近的模型状态
                if closest_state:
                    # Load the closest model for prediction
                    print("Retrieval")
                    retrieval_model = Batch_MLPModel(num_features, num_classes)
                    retrieval_model.load_state_dict(closest_state)
                    predicted_labels = predict_batch_model(retrieval_model, features)
                    accuracy = (predicted_labels == labels).float().mean()
                    accuracies.append(accuracy)
                    #if accuracies:
                    #    print("Accuracies collected so far:", accuracies)  # 打印目前收集的准确率列表
                    #print("Retrieval Shift handled.")
                elif last_labeled_data is not None:
                    print("Clustering")
                    #print("Features shape:", features.numpy().shape)
                    #print("Features dtype:", features.numpy().dtype)
                   # print("Last shape:", last_labeled_data.numpy().shape)
                    #print("Last dtype:", last_labeled_data.numpy().dtype)
                    combined_data = torch.vstack((last_labeled_data.clone().detach(), features))

                    clusterlabel, _ = kmeans_torch(combined_data, num_clusters=num_classes)
                    new_labels_clusters = clusterlabel[-features.size(0):]
                    label_map = {i: mode(last_labels.numpy()[clusterlabel[:50] == i], keepdims=True)[0][0] for i in range(num_classes)}
                    predicted_labels = np.array([label_map[cluster.item()] for cluster in new_labels_clusters])
                    if predicted_labels.shape == labels.numpy().shape:
                        accuracy = (predicted_labels == labels.numpy()).astype(float).mean()
                    else:
                        print("Shape mismatch:", predicted_labels.shape, labels.numpy().shape)
                        accuracy = 0  # 或者处理形状不匹配的情况
                    accuracies.append(accuracy)
                else:
                    print("First Severe")
                    # Ensemble predictions from both models
                    pred_inc = predict_incremental(incremental_model, features)
                    pred_batch = predict_batch_model(batch_model, features)
                    weights_inc = gaussian_kernel(shift_distance)
                    weights_batch = gaussian_kernel(shift_distance)
                    ensemble_pred = (weights_inc * pred_inc + weights_batch * pred_batch) / (weights_inc + weights_batch)
                    accuracy = (ensemble_pred == labels).float().mean()
                    accuracies.append(accuracy)
                    #if accuracies:
                    #    print("Accuracies collected so far:", accuracies)  # 打印目前收集的准确率列表
                    #print("First severe handled.")

            elif shift_type == "Slight shift":
                print("Slight")
                # Ensemble predictions from both models
                pred_inc = predict_incremental(incremental_model, features)
                if adaptive_window.saved_features is not None:
                    pred_batch = predict_batch_model(batch_model, features)
                    weights_inc = gaussian_kernel(shift_distance)
                    print("Weights from incremental model:", weights_inc)

                    saved_features_tensor = adaptive_window.saved_features
                    batch_shift_distance = dist_calc.calculate_shift_distance(features.numpy(), saved_features_tensor)
                    print(shift_distance)
                    print(batch_shift_distance)
                    weights_batch = gaussian_kernel(batch_shift_distance)
                    print("Weights from batch model:", weights_batch)

                    ensemble_pred = (weights_inc * pred_inc + weights_batch * pred_batch) / (weights_inc + weights_batch)
                    accuracy = (ensemble_pred == labels).float().mean()
                    print("Accuracy:", accuracy)
                    accuracies.append(accuracy)
                else:
                    accuracy = (pred_inc == labels).float().mean()
                    accuracies.append(accuracy)


                #if accuracies:
                 #   print("Accuracies collected so far:", accuracies)  # 打印目前收集的准确率列表
                #print("Slight Shift handled.")

        # Always train the incremental model
        train_incremental(incremental_model, features, labels, optimizer_incremental, criterion)
        #print("Incremental train")
        last_batch_data = features
        if labels.size(0) >= 50:
            last_labeled_data = features[-50:]
            last_labels = labels[-50:]
        # Update the batch model conditionally
        adaptive_window.add_batch(batch_data.numpy())
        if adaptive_window.should_update():
            #print("batch train")
            train_batch_model(batch_model, adaptive_window, optimizer_batch, criterion)
            adaptive_window.save_current_features()  # Save the current state before update
            adaptive_window.clear_window()
            model_history.add_model(batch_model.state_dict(), features.numpy())


    # Save accuracies to a file
    if accuracies:
        print("Accuracies collected so far:", accuracies)  # 打印目前收集的准确率列表
    np.savetxt("accuracies.txt", np.array(accuracies))

if __name__ == "__main__":
    main()
