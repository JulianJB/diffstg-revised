# based on: train_128_debug_forearm
import os
import torch
import argparse
import numpy as np
import torch.utils.data
from easydict import EasyDict as edict
from timeit import default_timer as timer

from utils.eval_128_bodysys25_0825 import Metric

from utils.common_utils import dir_check, to_device, ws, unfold_dict, dict_merge, Logger

from algorithm.dataset_128_bodysys25_0825 import HARDataset
from algorithm.diffstg.model_128_bodysys25_0825 import DiffSTG, save2file

from torch.utils.data import DataLoader

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

# for tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard'))
except:
    pass

def get_params():
    parser = argparse.ArgumentParser(description='Entry point of the code')

    # model
    parser.add_argument("--epsilon_theta", type=str, default='UGnet')
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--beta_schedule", type=str, default='quad')  # uniform, quad
    parser.add_argument("--beta_end", type=float, default=0.1)
    parser.add_argument("--sample_steps", type=int, default=200)  # sample_steps
    parser.add_argument("--ss", type=str, default='ddpm') #help='sample strategy', ddpm, multi_diffusion, one_diffusion
    # for imputation tasks (or generation tasks) set T_h (history) to the entire length of the window
    parser.add_argument("--T_h", type=int, default=128)

    # evaluation
    parser.add_argument('--n_samples', type=int, default=8)

    # train
    parser.add_argument("--is_train", type=bool, default=True) # train or evaluate
    parser.add_argument("--data", type=str, default='HAR')
    parser.add_argument("--mask_ratio", type=float, default=0.0) # mask of history data (not using this)
    parser.add_argument("--is_test", type=bool, default=False) # set to True for quick debugging
    parser.add_argument("--nni", type=bool, default=False) # not using this
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=16) # use this batch size for RACE (A10G)
    #parser.add_argument("--batch_size", type=int, default=4) # use this batch size for local GPU
    args, _ = parser.parse_known_args()
    return args

def default_config(data='HAR'):
    config = edict()

    # paths for forearm position, change for thigh or whatever other needed
    config.PATH_MOD = ws + '/output/HAR_forearm/model/'
    config.PATH_LOG = ws + '/output/HAR_forearm/log/'
    config.PATH_IMPUTATION = ws + '/output/HAR_forearm/imputations/'

    # Data Config
    config.data = edict()
    config.data.name = data
    config.data.path = ws + '/data/dataset/'

    # get the features from the training file
    config.data.feature_file = config.data.path + config.data.name + '/raw_train_data_forearm.npy'
    # get the adjacency matrix from file
    config.data.spatial = config.data.path + config.data.name + '/adj.npy'
    config.data.num_recent = 1

    # Initialize config.model earlier
    config.model = edict()

    if config.data.name == 'HAR':
        config.data.num_features = 3 # number of accelerometers
        config.data.num_vertices = 7 # number of positions
        # T_h and T_p are critical (e.g., history of 24 samples, predict 1 or more)
        config.model.T_h = 128  # Example, adjust as needed
        config.model.T_p = 0  # for imputing concurrent step

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('torch.cuda.is_available():', torch.cuda.is_available()) # verify running on GPU
    print('torch.cuda.device_count():', torch.cuda.device_count())
    print('torch.cuda.current_device():', torch.cuda.current_device())
    print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))

    config.model.V = config.data.num_vertices
    config.model.F = config.data.num_features

    config.model.device = device
    config.model.d_h = 32 # capacity of the model (hidden size, decrease for speed, increase for robustness)

    # config for diffusion model
    config.model.N = 200
    config.model.sample_steps = 200 # training steps
    config.model.epsilon_theta = 'UGnet'
    config.model.is_label_condition = True
    #config.model.beta_end = 0.02
    config.model.beta_end = 0.1 # as in original paper I think
    config.model.beta_schedule = 'quad'
    config.model.sample_strategy = 'ddpm'

    # In default_config function
    config.evaluation = edict()
    config.evaluation.val_sample_steps = 20  # sampling steps (validation)
    config.evaluation.val_sample_strategy = 'ddim_multi'

    config.n_samples = 8 # number of samples to be generated, defaults to 8

    # config for UGnet
    config.model.channel_multipliers = [1, 2] # the list of channel numbers at each resolution
    config.model.supports_len = 2

    # training config
    config.model_name = 'DiffSTG'
    config.is_test = False  # whether run the code in the test mode
    config.epoch = 20  # max number of training epochs
    #config.epoch = 5  # max number of training epochs supported by the local GPU (NVIDIA RTX 3050 Laptop GPU)
    config.optimizer = "adam"
    config.lr = 0.002 # learning rate
    #config.lr = 0.001

    config.batch_size = 16
    #config.batch_size = 4 # max batch size supported by the local GPU (NVIDIA RTX 3050 Laptop GPU)
    config.wd = 1e-5
    #config.early_stop = 10
    config.early_stop = 5
    config.start_epoch = 0
    config.device = device
    config.logger = Logger()

    if not os.path.exists(config.PATH_MOD):
        os.makedirs(config.PATH_MOD)
    if not os.path.exists(config.PATH_LOG):
        os.makedirs(config.PATH_LOG)
    if not os.path.exists(config.PATH_IMPUTATION):
        os.makedirs(config.PATH_IMPUTATION)
    return config

# A function for handling the imputation task via the validation step
def evals(model, data_loader, epoch, main_metric_obj, config, scaler_for_denorm, mode='test', train_log_prefix=""):
    setup_seed(2022)
    model.eval()

    # Lists to store results from all batches
    all_gt_denorm_list = []
    all_pred_denorm_multi_sample_list = [] # list to store all the samples generated by the model

    with torch.no_grad():  # Disable gradient calculations for inference
        for i, batch in enumerate(data_loader):
            # The dataloader now returns (ground_truth_data, data_to_be_masked)
            gt_data_norm, cond_data_norm = to_device(batch, config.device)

            # Ground truth (original scale will be recovered after denorm)
            x0_gt_norm = gt_data_norm.clone()  # (B, T_h, V, F)

            # Conditional input for model (mask 'position')
            # We assume that the target position was added at the end of each subject in the dataset
            x_cond_norm = cond_data_norm.clone()
            target_node_idx = config.model.V - 1  # Assuming target is the last sensor
            x_cond_norm[:, :, target_node_idx, :] = 0  # Mask the target sensor

            # Transpose condition for model input
            # Reshape for model input: (B, T_h, V, F) -> (B, F, V, T_h)
            x_cond_norm_t = x_cond_norm.transpose(1, 3)  # (B, F, V, T_h)

            # Model prediction (normalized scale)
            n_eval_samples = config.n_samples  # Or getattr for n_samples
            pred_norm_t = model(x_cond_norm_t, n_samples=n_eval_samples)  # Shape: (B, S, F, V, T_h)

            # Denormalize All N samples
            # Permute pred_norm_t from (B, S, F, V, T_h) to (B, S, T_h, V, F) for scaler
            pred_for_denorm = pred_norm_t.permute(0, 1, 4, 3, 2) # B, S, T_h, V, F
            pred_for_denorm_np = pred_for_denorm.cpu().numpy()

            # Inverse transform both predictions and ground truth to their original scale
            pred_denorm_samples = scaler_for_denorm.inverse_transform(pred_for_denorm_np)
            gt_denorm = scaler_for_denorm.inverse_transform(x0_gt_norm.cpu().numpy())

            # Append to the list for later concatenation
            all_pred_denorm_multi_sample_list.append(pred_denorm_samples)
            all_gt_denorm_list.append(gt_denorm)

    # Concatenate results from all batches into single numpy arrays
    final_gt_denorm = np.concatenate(all_gt_denorm_list, axis=0)
    final_pred_denorm_multi_sample = np.concatenate(all_pred_denorm_multi_sample_list, axis=0)
    # Shape of final_pred_denorm_multi_sample: (Total_Windows, S, T_h, V, F)

    # Clip all individual samples (important for MIS/CRPS if they are sensitive to extremes)
    final_pred_denorm_multi_sample_clipped = np.clip(final_pred_denorm_multi_sample,
                                                     a_min=config.data.clip_min,
                                                     a_max=config.data.clip_max)

    # Now call update_metrics function with the full multi-sample (and clipped) predictions
    # Metric.update_metrics will handle calculating CRPS/MIS on this multi-sample input
    # and then take the mean internally for MAE/RMSE
    main_metric_obj.update_metrics(final_gt_denorm, final_pred_denorm_multi_sample_clipped)
    main_metric_obj.update_best_metrics(epoch=epoch)

    if mode == 'validation':
        val_mae = main_metric_obj.metrics.get('mae', float('nan'))
        val_rmse = main_metric_obj.metrics.get('rmse', float('nan'))
        val_time = main_metric_obj.metrics.get('time', float('nan'))

        # Format matches the table header: "MAE    RMSE   Time"
        message = f" | {val_mae:<8.4f} {val_rmse:<8.4f} {val_time:<6.1f}s"

        # For console: append to the training progress line
        print(message, end='')

        # For logger: combine with the training part and add newline
        full_log_line = train_log_prefix + message + "\n"
        config.logger.write(full_log_line, is_terminal=False)  # Write to log file only, console is handled

    # For saving:
    if mode == 'test':
        print('final_gt_denorm.shape', final_gt_denorm.shape)
        print('final_pred_denorm_multi_sample_clipped.shape', final_pred_denorm_multi_sample_clipped.shape)

        # Get the index of the target position ('thigh')
        # Assuming 'thigh' is the last node, and config.model.V is the total number of nodes (7)
        target_node_idx = config.model.V - 1  # This will be 6 if V=7

        target_sensor_name = "target_sensor"  # Generic name

        # Isolate the ground truth for the target sensor
        gt_target_sensor_data = final_gt_denorm[:, :, target_node_idx, :]

        # Isolate the generated samples for the target sensor
        pred_target_sensor_samples = final_pred_denorm_multi_sample_clipped[:, :, :, target_node_idx, :]

        # Calculate the mean prediction for a single-point estimate comparison
        pred_target_sensor_mean = np.mean(pred_target_sensor_samples, axis=1)

        print(f"Shape of GT for '{target_sensor_name}': {gt_target_sensor_data.shape}")
        print(f"Shape of Mean Prediction for '{target_sensor_name}': {pred_target_sensor_mean.shape}")
        print(f"Shape of All Samples for '{target_sensor_name}': {pred_target_sensor_samples.shape}")

        # Define file paths for saving
        # check config.imputation_path points to a directory
        dir_check(config.imputation_path)  # Make sure the directory exists

        gt_filename = f"{config.imputation_path}/gt_{target_sensor_name}.npy"
        pred_mean_filename = f"{config.imputation_path}/pred_mean_{target_sensor_name}.npy"
        pred_samples_filename = f"{config.imputation_path}/pred_samples_{target_sensor_name}.npy"

        # save the NumPy arrays
        try:
            np.save(gt_filename, gt_target_sensor_data)
            np.save(pred_mean_filename, pred_target_sensor_mean)
            np.save(pred_samples_filename, pred_target_sensor_samples)

            save_message = (
                f"Saved '{target_sensor_name}' imputation results:\n"
                f"  GT: {gt_filename}\n"
                f"  Pred (Mean): {pred_mean_filename}\n"
                f"  Pred (Samples): {pred_samples_filename}\n"
            )
            config.logger.message_buffer += save_message

        except Exception as e:
            print(f"Error saving imputation results: {e}")
            config.logger.write(f"Error saving imputation results: {e}\n", is_terminal=True)

    model.train()
    return main_metric_obj

# Functions for data normalization
class StandardScalerTimeSeries:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """
        Calculates mean and std from the provided data.
        Data shape: (num_samples, num_nodes, num_features)
        Calculates statistics globally or feature-wise.
        For HAR, feature-wise (per accelerometer axis)
        """

        # Feature-wise mean and std (Recommended for HAR)
        # Assumes data shape (samples, nodes, features)
        self.mean = np.mean(data, axis=(0, 1), keepdims=True) # Shape (1, 1, num_features)
        self.std = np.std(data, axis=(0, 1), keepdims=True)   # Shape (1, 1, num_features)

        # Handle cases where std might be zero (e.g., if a feature is constant)
        self.std[self.std == 0] = 1e-6
        print(f"Calculated Norm Stats - Mean: {self.mean.squeeze()}, Std: {self.std.squeeze()}")

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fit yet.")
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data_scaled):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fit yet.")
        return (data_scaled * self.std) + self.mean

from pprint import  pprint
def main(params: dict):
    # torch.manual_seed(2022)
    setup_seed(2022)
    torch.set_num_threads(2)
    config = default_config(params['data'])
    config.is_test = params['is_test']

    config.nni = params.get('nni', False)  # Propagate the nni flag to config if functions read from config

    config.lr = params['lr']
    config.batch_size = params['batch_size']
    config.mask_ratio = params['mask_ratio']

    # model
    config.model.N = params['N']
    config.T_h = 128
    config.T_p = 0
    config.model.epsilon_theta =  params['epsilon_theta']
    config.model.sample_steps = params['sample_steps']
    config.model.d_h = params['hidden_size']
    config.model.C = params['hidden_size']
    config.model.n_channels = params['hidden_size']
    config.model.beta_end = params['beta_end']
    config.model.beta_schedule = params["beta_schedule"]
    config.model.sample_strategy = params["ss"]
    config.n_samples = params['n_samples']

    if config.model.sample_steps > config.model.N:
        print('sample steps large than N, exit')
        #nni.report_intermediate_result(50)
        if config.nni:  # Use the flag from config (or params directly)
            import nni
            nni.report_final_result(50)
        return 0

    config.trial_name = '+'.join([f"{v}" for k, v in params.items()])
    config.log_path = f"{config.PATH_LOG}/{config.trial_name}.log"

    pprint(config)
    dir_check(config.log_path)
    config.logger.open(config.log_path, mode="w")
    #log parameters
    config.logger.write(config.__str__()+'\n', is_terminal=False)

    config.imputation_path = imputation_path = config.PATH_IMPUTATION
    config.logger.write(f"imputation_path directory:{imputation_path}\n", is_terminal=False)
    print('imputation_path directory:', imputation_path)
    dir_check(imputation_path)

    # Define the path to the HAR results/data directory
    har_data_path = './data/dataset/HAR/'

    # Load the adjacency matrix for the REALWORLD HAR dataset
    adjacency_matrix_har = np.load(f'{har_data_path}/adj.npy')

    # Assign it to the config object where the model expects it
    config.model.A = adjacency_matrix_har

    # Minimum and Maximum values for the REALWORLD HAR dataset readings
    config.data.clip_min = np.array([-19.6091, -19.9123, -19.6085])
    config.data.clip_max = np.array([19.6079, 19.6085, 19.6085])

    print(f"Loaded HAR adjacency matrix with shape: {config.model.A.shape}")

    #train_data = np.load(f'{har_data_path}/raw_training_data.npy').astype(np.float32) # use this one for thigh
    train_data = np.load(f'{har_data_path}/raw_train_data_forearm.npy').astype(np.float32) # use this one for forearm

    #val_data = np.load(f'{har_data_path}/raw_val_data.npy').astype(np.float32) # use this one for thigh
    val_data = np.load(f'{har_data_path}/raw_val_data_forearm.npy').astype(np.float32) ## use this one for forearm

    #test_data = np.load(f'{har_data_path}/raw_test_data.npy').astype(np.float32) # use this one for thigh
    test_data = np.load(f'{har_data_path}/raw_test_data_forearm.npy').astype(np.float32) # use this one for forearm

    print('train_data.shape', train_data.shape)
    print('val_data.shape', val_data.shape)
    print('test_data.shape', test_data.shape)

    actual_train_samples = train_data
    actual_val_samples = val_data
    test_array_this_fold = test_data

    # Normalize Data for the Current Fold
    scaler_fold = StandardScalerTimeSeries()

    # Fit the scaler on the actual training data of this fold
    train_data_normalized_fold = scaler_fold.fit_transform(actual_train_samples)

    # Transform validation and test data using the SAME scaler (fitted on train)
    val_data_normalized_fold = scaler_fold.transform(actual_val_samples)
    test_data_normalized_fold = scaler_fold.transform(test_array_this_fold)

    # Instantiate HARDataset with Normalized Data
    train_dataset_fold = HARDataset(data_array=train_data_normalized_fold,
                                    config=config,
                                    dataset_name=f'HAR_train_fold')

    val_dataset_fold = HARDataset(data_array=val_data_normalized_fold,
                                  config=config,
                                  dataset_name=f'HAR_val_fold')

    test_dataset_fold = HARDataset(data_array=test_data_normalized_fold,
                                   config=config,
                                   dataset_name=f'HAR_test_fold')

    train_loader = DataLoader(train_dataset_fold, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_fold, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset_fold, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda") # force CUDA as the device for training
    model = DiffSTG(config.model)
    #model = model.to(config.device)
    model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # TODO: Option A (Conservative LR adjustment) use with LR = 0.002
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6)
    # TODO: Option B (More Proactive LR adjustment) use with LR = 0.001 (or 0.0005)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    model_path = config.PATH_MOD + config.trial_name + model.model_file_name()
    config.model_path = model_path
    config.logger.write(f"model path:{model_path}\n", is_terminal=False)
    print('model_path:', model_path)
    dir_check(model_path)

    config.imputation_path = imputation_path = config.PATH_IMPUTATION
    config.logger.write(f"imputation_path:{model_path}\n", is_terminal=False)
    print('imputation_path:', imputation_path)
    dir_check(imputation_path)

    # log model architecture
    print(model)
    config.logger.write(model.__str__())

    best_val_metric = float('inf')  # Assuming lower is better (e.g., MAE/RMSE)
    best_epoch = -1

    # log training process
    config.logger.write(f'Num_of_parameters:{sum([p.numel() for p in model.parameters()])}\n', is_terminal=True)
    #message = "      |---Train--- |---Val Future-- -|-----Val History----|\n"
    print('')
    message = "      |  ---Train---   |   ----Validation---   | (Metrics over T_h window)\n" # formatting
    config.logger.write(message, is_terminal=True)

    #message = "Epoch | Loss  Time | MAE     RMSE    |  MAE    RMSE   Time|\n" #f"{'Type':^5}{'Epoch':^5} | {'MAE':^7}{'RMSE':^7}{'MAPE':^7}
    message = "Epoch |   Loss  Time   |   MAE    RMSE   Time  |\n"  # f"{'Type':^5}{'Epoch':^5} | {'MAE':^7}{'RMSE':^7}{'MAPE':^7}
    config.logger.write(message, is_terminal=True)


    train_start_t = timer()
    #print('Training sample steps: ', config.model.sample_steps)
    # Train and sample the data
    for epoch in range(config.epoch):
        if not params['is_train']: break
        #if epoch > 1 and config.is_test: break # TODO: Enabled in the original impl.

        n, avg_loss, time_lst = 0, 0, []

        model.train() # Ensure model is in training mode
        n_batches_processed_in_epoch = 0 # To handle avg_loss correctly if loop breaks early
        cumulative_loss_in_epoch = 0.0
        epoch_time_lst = [] # For total epoch training time

        # train diffusion model
        for i, batch in enumerate(train_loader):
            #if i > 3: break  # For quick testing
            #if i > 3 and config.is_test: break # For quick testing
            time_start_batch = timer()

            # If T_p = 0, future_data will be a tensor with a 0 in its time dimension.
            # history_data will be (B, T_h, V, F)
            ground_truth_data, data_to_be_masked = batch

            # Ground truth x0 for the diffusion model
            # This is the complete window to reconstruct
            x0_gt = ground_truth_data.to(config.device)  # Shape: (B, T_h, V, F)

            # Conditional input (x_cond)
            # This is the history window with the target position ('thigh') masked out
            x_cond_input = data_to_be_masked.clone().to(config.device)
            target_node_idx = config.model.V - 1 # Assuming 'thigh' is the last node
            x_cond_input[:, :, target_node_idx, :] = 0 # Mask all time steps and features of 'thigh'

            x = x0_gt.clone() # This variable name matches original loop's `x` if T_p=0
            x_masked = x_cond_input.clone() # This variable name matches original loop's `x_masked` if T_p=0

            # Ensure the data tensors for the model are float32
            x = x.float() # Converts x to torch.float32
            x_masked = x_masked.float() # Converts x_masked to torch.float32

            # reshape, required for the Temporal Convolution Network (TCN) and Graph Convolutional Networks (GCNs) layers
            x = x.transpose(1,3) # (B, F, V, T) -> batch_size, features, vertices, timesteps
            x_masked = x_masked.transpose(1,3) # (B, F, V, T) -> batch_size, features, vertices, timesteps

            # Loss calculation
            # TODO: some notes added:
            # The multiplication by 10 in the line loss = 10 * model.loss(...) is a loss scaling hyperparameter
            # The DiffSTG paper does not explicitly mention or justify this specific value.
            # This means it is not a constant derived from the fundamental theory of Denoising Diffusion Probabilistic Models (DDPMs).
            # Instead, it is an empirical choice made by the authors during implementation.
            loss = 10 * model.loss(x, x_masked)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping here for training stability
            # TODO: add code for gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # calculate the moving average training loss
            n += 1
            avg_loss = avg_loss * (n - 1) / n + loss.item() / n

            # Update per-batch stats for console
            n_batches_processed_in_epoch += 1
            cumulative_loss_in_epoch += loss.item()
            current_avg_loss_for_console = cumulative_loss_in_epoch / n_batches_processed_in_epoch

            epoch_time_lst.append((timer() - time_start_batch))
            current_cumulative_time_for_console = np.sum(epoch_time_lst)

            # CONSOLE ONLY: Live update for training progress
            console_train_message = f"{epoch + (i + 1) / len(train_loader):6.1f}| {current_avg_loss_for_console:0.3f} {current_cumulative_time_for_console:.1f}s"
            print('\r' + console_train_message, end='', flush=True)

        final_avg_loss_epoch = cumulative_loss_in_epoch / n_batches_processed_in_epoch if n_batches_processed_in_epoch > 0 else float(
            'nan')
        final_epoch_train_time = np.sum(epoch_time_lst)

        # the training part of the log line (to be combined with validation)
        train_log_part = f"{epoch + 1:6.1f}    {final_avg_loss_epoch:5.3f} {final_epoch_train_time:6.1f}s"

        print('\r' + train_log_part, end='', flush=True) # Print the final training part for this epoch

        try:
            writer.add_scalar('train/loss', avg_loss, epoch)
        except:
            pass

        metrics_val = Metric(T_p=config.model.T_h) # Evaluate over T_h window for imputation tasks

        if epoch >= config.start_epoch:
            original_model_sample_steps = config.model.sample_steps
            original_model_strategy = config.model.sample_strategy

            # Use fewer steps for validation
            val_sample_steps = getattr(config.evaluation, 'val_sample_steps', 20)
            val_sample_strategy = getattr(config.evaluation, 'val_sample_strategy', 'ddim_multi')

            config.model.sample_steps = val_sample_steps
            config.model.sample_strategy = val_sample_strategy
            if hasattr(model, 'set_ddim_sample_steps'):
                model.set_ddim_sample_steps(val_sample_steps)
            if hasattr(model, 'set_sample_strategy'):
                model.set_sample_strategy(val_sample_strategy)

            # Pass the train_log_part to evals so it can build the full log line
            evals(model, val_loader, epoch, metrics_val, config, scaler_fold, mode='validation',
                  train_log_prefix=train_log_part)

            # Restore original config values if they are globally used by other parts,
            # though the final test loop explicitly sets its own
            config.model.sample_steps = original_model_sample_steps
            config.model.sample_strategy = original_model_strategy
            # Also reset on the model object if these settings persist on it
            if hasattr(model, 'set_ddim_sample_steps'):
                model.set_ddim_sample_steps(original_model_sample_steps)
            if hasattr(model, 'set_sample_strategy'):
                model.set_sample_strategy(original_model_strategy)

        print()  # Newline after epoch line
        current_val_mae = metrics_val.metrics.get('mae', float('inf'))
        if current_val_mae < best_val_metric:
            best_val_metric = current_val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), model_path) # Save best model based on validation
            print('')
            print(f"[save model based on val_mae: {best_val_metric:.3f}] >> {model_path}", end='')
        else:
            torch.save(model.state_dict(), model_path)
            print('')
            print(f"[save model] >> {model_path}", end='')

        print()
        scheduler.step(current_val_mae)

        if epoch - best_epoch > config.early_stop: break # Early_stop

        train_end_t = timer()
        config.logger.write(f"Total training time: {train_end_t - train_start_t:.2f}s\n")

    try:
        model = DiffSTG(config.model)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.to(config.device)
        print('Best model state_dict loaded from: <<', model_path)
    except Exception as err:
        print(err)
        print('load best model failed')

    # conduct multiple-samples, then report the best
    metric_lst = []
    for sample_strategy, sample_steps in [('ddim_multi', 40)]:
        if sample_steps > config.model.N: break

        config.model.sample_strategy = sample_strategy
        config.model.sample_steps = sample_steps

        model.set_ddim_sample_steps(sample_steps)
        model.set_sample_strategy(sample_strategy)

        metrics_test = Metric(T_p=config.model.T_h)

        evals(model, test_loader, best_epoch, metrics_test, config, scaler_fold, mode='test')
        print('')
        message = f'sample_strategy: {sample_strategy}, sample_steps: {sample_steps}\nFinal results in test: {metrics_test}\n'
        config.logger.write(message, is_terminal=True)

        params = unfold_dict(config)
        params = dict_merge([params, metrics_test.to_dict()])
        params['best_epoch'] = metrics_val.best_metrics['epoch']
        params['model'] = config.model.epsilon_theta
        save2file(params)
        metric_lst.append(metrics_test.metrics['mae'])

    # rename log file
    log_file, log_name = os.path.split(config.log_path)
    new_log_path = os.path.join(log_file, f"[{config.data.name}]mae{min(metric_lst):9.4f}+{log_name}")
    import shutil
    shutil.copy(config.log_path, new_log_path)
    config.log_path = new_log_path

    try:
        writer.close()
    except:
        pass

    final_metric_to_report = min(metric_lst)
    print(f"Final metric for this run: {final_metric_to_report}")

    if config.nni:
        import nni  # Import nni only when needed
        nni.report_final_result(final_metric_to_report)

# In train.py
if __name__ == '__main__':
    import logging

    logger = logging.getLogger('training')

    # Start with parameters from get_params (CLI args or defaults)
    params = vars(get_params())

    # Try to get NNI parameters if running in an NNI environment
    try:
        import nni # Keep the import here for NNI runs

        tuner_params = nni.get_next_parameter()
        logger.info(f"NNI parameters received: {tuner_params}")
        params.update(tuner_params) # Override with NNI parameters
    except ImportError:
        logger.info("NNI module not found. Running with parameters from CLI or defaults.")
    except Exception as e: # Catches if nni.get_next_parameter() fails (e.g., not in NNI trial)
        logger.info(f"Not running in an NNI trial or NNI error ({e}). Using parameters from CLI or defaults.")

    try:
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise