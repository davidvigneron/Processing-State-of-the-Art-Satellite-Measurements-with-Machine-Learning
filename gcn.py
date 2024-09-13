#!./.venv/bin/python

#SBATCH --nodes=1
##SBATCH --ntasks=16
##SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
##SBATCH --mem=501600
#SBATCH --mail-type=END
##SBATCH --mail-user=youremail@kit.edu

#SBATCH --time=0-06:00:00

#SBATCH --error=error_gcn.log

#SBATCH --output=output_gcn.log
#SBATCH --job-name=gcn


import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import r2_score
import numpy as np

from path_constants import *
import logging

logger = logging.getLogger(__name__)


# if MULTI_OPTIM is True, assume that 4 GPUs are available and train different models on those 4 GPUs in parallel
MULTI_OPTIM = False
# RELOAD reloads the state dict of the previous model, I used it to fine tune the model with lower lr
RELOAD = False
PLOT = True

# Hyperparameters
THRESH_TIME = 137438950000 * 20 # 137438950000 is about the distance between 2 observations not happening at the exact same time
THRESH_SPACE = 10
HIDDEN_CHANNELS = 512
LEARNING_RATE = .00015 if not RELOAD else .00001
EPOCHS = 100 if not RELOAD else 200
BATCH_SIZE = 45000

MODEL_NUMBER = 4



torch.manual_seed(9)
    
class SatelliteGCN(nn.Module):
    def __init__(self, node_features, hidden_channels, out_features, k=2):
        super(SatelliteGCN, self).__init__()
        # Use ModuleList to store convolutional layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_channels))
        if k > 1:
            self.convs.extend([GCNConv(hidden_channels, hidden_channels) for _ in range(k-2)])
        self.out = nn.Linear(hidden_channels, out_features)
    def forward(self, x, edge_index, edge_attr=None):
        # Apply all the GCNConv layers in the convs list
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
        x = self.out(x)
        return x.squeeze(-1)
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def create_graph(X, y, THRESH_TIME, THRESH_SPACE, device):
    # last three columns of X are time, lat, lon
    time, lat, lon = X[:, -3].to(device), X[:, -2].to(device), X[:, -1].to(device)
    features = X[:, :-3]
    # Create edge_index
    coords = torch.stack([lat, lon], dim=1)
    dist_space = torch.cdist(coords, coords)
    dist_time = torch.abs(time.unsqueeze(1) - time.unsqueeze(0))
    edge_index_space_and_time = ((dist_space < THRESH_SPACE) & (dist_time < THRESH_TIME)).nonzero(as_tuple=False).t()
    edge_index = torch.unique(edge_index_space_and_time, dim=1)
    edge_index= edge_index[:, edge_index[0] != edge_index[1]]
    # Create edge_attr as sum of the two distances
    src, dst = edge_index
    dist_space = torch.sqrt((lat[src] - lat[dst])**2 + (lon[src] - lon[dst])**2)
    dist_time = torch.abs(time[src] - time[dst])
    edge_attr = (dist_space + dist_time).view(-1, 1)
    logger.debug(f"Number of edges: {edge_index.shape[1]} / {time.shape[0]}")
    return Data(x=features, edge_index=edge_index.detach(), edge_attr=edge_attr.detach(), y=y).detach()

def create_dataloaders(X_train, Y_train, X_test, Y_test, THRESH_TIME, THRESH_SPACE, batch_size=32, device="cpu"):
    train_graphs = [create_graph(X_train[i:i+batch_size], Y_train[i:i+batch_size], THRESH_TIME, THRESH_SPACE, device) 
                    for i in range(0, len(X_train), batch_size)]
    test_graphs = [create_graph(X_test[i:i+batch_size], Y_test[i:i+batch_size], THRESH_TIME, THRESH_SPACE, device) 
                   for i in range(0, len(X_test), batch_size)]
    
    train_loader = train_graphs
    test_loader = test_graphs
    
    return train_loader, test_loader

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for graph in train_loader:
        optimizer.zero_grad()
        graph = graph.to(device)
        out = model(graph.x, graph.edge_index, graph.edge_attr)
        loss = criterion(out, graph.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        graph = graph.to("cpu")
        torch.cuda.empty_cache()
    return total_loss / len(train_loader)

def test(model, test_loader, criterion, device, plot=False):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for graph in test_loader:
            graph = graph.to(device)
            out = model(graph.x, graph.edge_index, graph.edge_attr)
            loss = criterion(out, graph.y)
            total_loss += loss.item()
            torch.cuda.empty_cache()
            graph = graph.to("cpu")
            predictions.extend(out.cpu().numpy())
            targets.extend(graph.y.cpu().numpy())

        predictions = np.stack(predictions)
        targets = np.stack(targets)
        r2: np.ndarray = r2_score(targets, predictions, multioutput="raw_values") # type: ignore


        if not MULTI_OPTIM and plot:
            import matplotlib.pyplot as plt
            import mpl_scatter_density
            from matplotlib.colors import LinearSegmentedColormap
            white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
                (0, '#ffffff'),
                (sys.float_info.epsilon, '#440053'),
                (0.2, '#404388'),
                (0.4, '#2a788e'),
                (0.6, '#21a784'),
                (0.8, '#78d151'),
                (1, '#fde624'),
            ], N=512)
            def plot_scatter(outputs, truth, var_name, r2, dimensions):
                fig, axes = plt.subplots(5, 6, figsize=(15, 12), subplot_kw={'projection': 'scatter_density'})  # 5 rows and 6 columns of subplots
                fig.suptitle(f'Comparison of predicted and true values for {var_name}')
                for i in range(dimensions):
                    row, col = divmod(i, 6)
                    ax = axes[row, col]
                    cb = ax.scatter_density(outputs[:,i], truth[:,i], cmap=white_viridis, dpi=600)
                    plt.colorbar(cb, ax=ax)
                    mean = np.mean(truth[:,i])
                    ax.axline((mean, mean), slope=1, c="black", linestyle="--", linewidth=.5, alpha=.5)
                    ax.set_title(f'{var_name} lvl {i}\nR2: {r2[i]:<.6}', fontsize='x-small')
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Full physics")
                # Hide any unused subplots
                for i in range(dimensions, 30):
                    fig.delaxes(axes.flatten()[i])
                plt.tight_layout(rect=(0, 0, 0.98, 1))
                plt.savefig(f"python/images/result_gcn_{var_name}_w{MODEL_NUMBER}_2019.pdf", dpi=600)



            plot_scatter(predictions[:,1:30], targets[:,1:30], "at", r2[1:30], 29)
            plot_scatter(predictions[:,30:], targets[:,30:], "wv", r2[30:], 29) 
            # Plotting for musica_st
            fig = plt.figure()
            ax = fig.add_subplot(projection='scatter_density')
            cb = ax.scatter_density(predictions[:,0], targets[:,0], cmap=white_viridis, dpi=300)  # type: ignore
            plt.colorbar(cb, ax=ax)
            mean = np.mean(predictions[:, 0])
            display_str = f"Comparison of predicted and true values for musica_st\nR2: {r2[0]:<.6}" 
            ax.axline((mean, mean), slope=1, c="black", linestyle="--", linewidth=1.0, alpha=.5) # type: ignore
            ax.set_xlabel("Predicted surface temp")
            ax.set_ylabel("Full physics surface temp")

            plt.title(display_str)
            plt.savefig(f"python/images/result_gcn_musica_st_w{MODEL_NUMBER}_2019.pdf") #, dpi=300)

    
    return total_loss / len(test_loader), r2

def main(X_train_norm, Y_train, X_test_norm, Y_test, device, optim_name="adam", lr=LEARNING_RATE, scheduler_name="cosine", num_layers=2, weight_decay=0.0, factor_thresh_time = 1., factor_thresh_space = 1.):
    # global LEARNING_RATE, THRESH_SPACE, THRESH_TIME, BATCH_SIZE, RELOAD, MULTI_OPTIM, EPOCHS

    train_loader, test_loader = create_dataloaders(X_train_norm, Y_train, X_test_norm, Y_test, THRESH_TIME*factor_thresh_time, THRESH_SPACE*factor_thresh_space, BATCH_SIZE, device)

    # Initialize model
    node_features = X_train_norm.shape[1] - 3  # Subtract 3 for time, lat, lon
    model_path = f"gnn_w{MODEL_NUMBER}_k{num_layers}{'c' if not MULTI_OPTIM else 'd'}.pt"

    model = SatelliteGCN(node_features, HIDDEN_CHANNELS, Y_train.shape[1], k=num_layers).to(device)
    if RELOAD:
        model.load_state_dict(torch.load(model_path))
    # Define optimizer and loss function
    import collections
    optimizers = collections.defaultdict(lambda: optim.Adam(model.parameters(), lr=lr if not RELOAD else 10, betas=(0.9, 0.999), amsgrad=False, weight_decay=weight_decay))
    
    for k, v in {  "adam":optim.Adam(model.parameters(), lr=lr, amsgrad=False, weight_decay=weight_decay),
            "sgd0": optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0),
            "sgd9": optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9),
            "adagrad": optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
            }.items():
        optimizers[k] = v

    
    optimizer = optimizers[optim_name]
    criterion = nn.MSELoss(reduction="sum")
    
    if scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=5)
    elif scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120, 160], gamma=0.5)
    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, r2 = test(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        logger.debug(f"Num layers: {num_layers} LR: {lr:<5}, {get_lr(optimizer):.5e} Scheduler: {scheduler_name:9}, factors thresh t,s: {factor_thresh_time:<2}, {factor_thresh_space:<2} Epoch {epoch+1:3}/{EPOCHS}, Train Loss: {train_loss:20.2f}, Test Loss: {test_loss:20.2f}, R2 Score: {r2.mean(axis=0)}")
        
    
    # Final evaluation
    _, final_r2 = test(model, test_loader, criterion, device, PLOT)

    torch.save(model.state_dict(), model_path)
    logger.info(f"Model state dict saved at {model_path}")

    print(f"Final R2 Score: {final_r2}")

    # plot losses during training
    import matplotlib.pyplot as plt
    (fig, axes) = plt.subplots(1,2)
    axes[0].plot(range(len(train_losses)), train_losses, label="train loss")
    axes[0].plot(range(len(test_losses)), test_losses, label="test loss")
    axes[0].legend()
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epoch")
    short_start = EPOCHS // 2
    axes[1].plot(range(short_start, len(train_losses)), train_losses[short_start:], label="train loss")
    axes[1].plot(range(short_start, len(test_losses)), test_losses[short_start:], label="test loss")
    axes[1].legend()
    axes[1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(f"python/images/gnn_loss{num_layers}_{weight_decay}.png")

    return final_r2



if __name__ == "__main__":
    skip_step = 100
    X_train = np.load(PATH_TO_2020_X_W_TIME, mmap_mode="r")[::skip_step,:].copy()
    Y_train = np.load(PATH_TO_2020_Y_W_TIME, mmap_mode="r")[::skip_step,:].copy()
    X_test = np.load(PATH_TO_2019_X_W_TIME, mmap_mode="r")[::skip_step,:].copy()
    Y_test = np.load(PATH_TO_2019_Y_W_TIME, mmap_mode="r")[::skip_step,:].copy()

    x_scaler = StandardScaler()
    x_scaler.fit(X_train[:,:-3])
    X_train[:,:-3] = x_scaler.transform(X_train[:,:-3])
    X_test[:,:-3] = x_scaler.transform(X_test[:,:-3])

    mean_y, std_y = Y_train.mean(axis=0), Y_train.std(axis=0)
    std_y[std_y == 0] = sys.float_info.epsilon
    Y_train_norm = (Y_train - mean_y) / std_y
    Y_test_norm = (Y_test - mean_y) / std_y

    import pickle
    scaler_file_name = "gcn_scaler.pkl"
    with open(scaler_file_name, "wb+") as path:
        pickle.dump((x_scaler, mean_y, std_y), path)
    logger.info(f"Scaler saved at {scaler_file_name}")


    if MULTI_OPTIM:
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        hparam = [1,2,3,4]
        futures = [executor.submit(main, torch.tensor(X_train), torch.tensor(Y_train_norm), torch.tensor(X_test), torch.tensor(Y_test_norm), device, num_layers=k, factor_thresh_time=.25, factor_thresh_space=.25) for device, k in zip([f"cuda:{i}" for i in range(4)] * 2, hparam)]
        res = [f.result() for f in futures]
        print("END:", res)
        print("means:", hparam, [r.mean() for r in res])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # torch.set_default_device(device)
        main(torch.tensor(X_train), torch.tensor(Y_train_norm), torch.tensor(X_test), torch.tensor(Y_test_norm), device, num_layers=4, factor_thresh_time=0, factor_thresh_space= 0)
