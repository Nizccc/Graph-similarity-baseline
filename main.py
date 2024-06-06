import os
import time
import glob

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from scipy.stats import spearmanr
from scipy.stats import kendalltau

from models import gnnbasedModel,mlpbasedModel,SAT_GSL
from parser_based import parameter_parser
from utils import tab_printer, GraphRegressionDataset, prec_at_ks, calculate_ranking_correlation, create_dir_if_not_exists

from datetime import datetime

args = parameter_parser()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
tab_printer(args)

create_dir_if_not_exists(args.log_path)
log_root_dir = args.log_path
signature = args.dataset + args.model + '@' + datetime.now().strftime("%Y-%m-%d@%H:%M:%S")
current_run_dir = os.path.join(log_root_dir, signature)
create_dir_if_not_exists(current_run_dir)
model_save_path = os.path.join(current_run_dir, 'best_model.pt')


def train():
    print('\nModel training.\n')
    start = datetime.now()
    val_loss_values = []
    patience_cnt = 0
    min_loss = 1e10

    with torch.autograd.detect_anomaly():
        for epoch in range(args.epochs):
            train_starttime = datetime.now()
            model.train()
            batches = dataset.create_batches(dataset.training_graphs)
            main_index = 0
            loss_sum = 0
            for index, batch_pair in enumerate(batches):
                optimizer.zero_grad()
                data = dataset.transform(batch_pair)

                if args.model == 'SAT_GSL':
                    prediction, L_cl= model(data)
                    loss = F.mse_loss(prediction, data['target'], reduction='sum') + L_cl
                else:
                    prediction = model(data)
                    loss = F.mse_loss(prediction, data['target'], reduction='sum')
                loss.backward()
                optimizer.step()
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss.item()
            loss = loss_sum / main_index

            time_spent = datetime.now() - train_starttime
            print('Epoch: {:05d},'.format(epoch), 'loss_train: {:.6f},'.format(loss), 'spent time: {}'.format(time_spent))

            if epoch >= args.iter_val_start and epoch % args.iter_val_every == 0 :
                val_loss,val_spenttime = validate()
                print('\tValEpoch: {:05d},'.format(epoch), 'loss_val: {:.6f},'.format(val_loss), 'spent time: {}'.format(val_spenttime))
                val_loss_values.append(val_loss)
                
                if val_loss_values[-1] < min_loss:
                    min_loss = val_loss_values[-1]
                    best_epoch = epoch
                    patience_cnt = 0
                    torch.save(model.state_dict(), model_save_path)
                else:
                    patience_cnt += 1

                if patience_cnt == args.patience:
                    break

        print('Optimization Finished! Total time elapsed: {}'.format(datetime.now() - start))


def validate():
    val_starttime = datetime.now()
    model.eval()
    batches = dataset.create_batches(dataset.val_graphs)
    main_index = 0
    loss_sum = 0
    with torch.no_grad():
        for index, batch_pair in enumerate(batches):
            data = dataset.transform(batch_pair)
            prediction = model(data)
            loss = F.mse_loss(prediction, data['target'], reduction='sum')
            main_index = main_index + batch_pair[0].num_graphs
            loss_sum = loss_sum + loss.item()
        loss = loss_sum / main_index

    return loss,datetime.now()-val_starttime


def evaluate():
    print('\nModel evaluation.')
    model.eval()
    scores = np.zeros((len(dataset.testing_graphs), len(dataset.trainval_graphs)))
    ground_truth = np.zeros((len(dataset.testing_graphs), len(dataset.trainval_graphs)))
    prediction_mat = np.zeros((len(dataset.testing_graphs), len(dataset.trainval_graphs)))

    rho_list = []
    tau_list = []
    prec_at_10_list = []
    prec_at_20_list = []

    with torch.no_grad():
        for i, g in enumerate(dataset.testing_graphs):
            if len(dataset.trainval_graphs) <= args.batch_size:
                source_batch = Batch.from_data_list([g] * len(dataset.trainval_graphs))
                target_batch = Batch.from_data_list(dataset.trainval_graphs)

                data = dataset.transform((source_batch, target_batch))
                target = data['target']
                ground_truth[i] = target.cpu().numpy()
                prediction = model(data)
                prediction_mat[i] = prediction.detach().cpu().numpy()

                scores[i] = F.mse_loss(prediction, target, reduction='none').detach().cpu().numpy()

                rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
                tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
                prec_at_10_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 10))
                prec_at_20_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 20))
            else:
                # Avoid GPU OOM error
                batch_index = 0
                target_loader = DataLoader(dataset.trainval_graphs, batch_size=args.batch_size, shuffle=False)
                for index, target_batch in enumerate(target_loader):
                    source_batch = Batch.from_data_list([g] * target_batch.num_graphs)
                    data = dataset.transform((source_batch, target_batch))
                    target = data['target']
                    num_graphs = target_batch.num_graphs
                    ground_truth[i,batch_index: batch_index+num_graphs] = target.cpu().numpy()
                    prediction = model(data)
                    prediction_mat[i,batch_index: batch_index+num_graphs] = prediction.detach().cpu().numpy()
                    scores[i,batch_index: batch_index+num_graphs] = F.mse_loss(prediction, target, reduction='none').detach().cpu().numpy()
                    batch_index += num_graphs
                
                rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
                tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
                prec_at_10_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 10))
                prec_at_20_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 20))

    rho = np.mean(rho_list)
    tau = np.mean(tau_list)
    prec_at_10 = np.mean(prec_at_10_list)
    prec_at_20 = np.mean(prec_at_20_list)
    model_error = np.mean(scores) 
    print_evaluation(model_error, rho, tau, prec_at_10 ,prec_at_20)
    test_results = {
    'mse': model_error,
    'rho': rho,
    'tau': tau,
    'prec_at_10': prec_at_10,
    'prec_at_20': prec_at_20
    }
    return test_results


def evaluate_crossdata():
    print('\nModel evaluation.')
    model.eval()
    scores = np.zeros((len(cross_dataset.testing_graphs), len(cross_dataset.trainval_graphs)))
    ground_truth = np.zeros((len(cross_dataset.testing_graphs), len(cross_dataset.trainval_graphs)))
    prediction_mat = np.zeros((len(cross_dataset.testing_graphs), len(cross_dataset.trainval_graphs)))

    rho_list = []
    tau_list = []
    prec_at_10_list = []
    prec_at_20_list = []

    with torch.no_grad():
        for i, g in enumerate(cross_dataset.testing_graphs):
            if len(cross_dataset.trainval_graphs) <= args.batch_size:
                source_batch = Batch.from_data_list([g] * len(cross_dataset.trainval_graphs))
                target_batch = Batch.from_data_list(cross_dataset.trainval_graphs)

                data = cross_dataset.transform((source_batch, target_batch))
                target = data['target']
                ground_truth[i] = target.cpu().numpy()
                prediction = model(data)
                prediction_mat[i] = prediction.detach().cpu().numpy()

                scores[i] = F.mse_loss(prediction, target, reduction='none').detach().cpu().numpy()

                rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
                tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
                prec_at_10_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 10))
                prec_at_20_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 20))
            else:
                # Avoid GPU OOM error
                batch_index = 0
                target_loader = DataLoader(cross_dataset.trainval_graphs, batch_size=args.batch_size, shuffle=False)
                for index, target_batch in enumerate(target_loader):
                    source_batch = Batch.from_data_list([g] * target_batch.num_graphs)
                    data = cross_dataset.transform((source_batch, target_batch))
                    target = data['target']
                    num_graphs = target_batch.num_graphs
                    ground_truth[i,batch_index: batch_index+num_graphs] = target.cpu().numpy()
                    prediction = model(data)
                    prediction_mat[i,batch_index: batch_index+num_graphs] = prediction.detach().cpu().numpy()
                    scores[i,batch_index: batch_index+num_graphs] = F.mse_loss(prediction, target, reduction='none').detach().cpu().numpy()
                    batch_index += num_graphs
                
                rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
                tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
                prec_at_10_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 10))
                prec_at_20_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 20))

    rho = np.mean(rho_list)
    tau = np.mean(tau_list)
    prec_at_10 = np.mean(prec_at_10_list)
    prec_at_20 = np.mean(prec_at_20_list)
    model_error = np.mean(scores) 
    print("crossdataset results：")
    print_evaluation(model_error, rho, tau, prec_at_10 ,prec_at_20)
    test_results = {
    'mse': model_error,
    'rho': rho,
    'tau': tau,
    'prec_at_10': prec_at_10,
    'prec_at_20': prec_at_20
    }
    return test_results


def print_evaluation(model_error, rho, tau, prec_at_10 ,prec_at_20):
    print("\nmse(10^-3): " + str(model_error * 1000))
    print("Spearman's rho: " + str(rho))
    print("Spearman's tau: " + str(tau))
    print("p@10: " + str(prec_at_10))
    print("p@20: " + str(prec_at_20))


if __name__ == "__main__":
    total_test_results = {
            'mse': 0,
            'rho': 0,
            'tau': 0,
            'prec_at_10': 0,
            'prec_at_20': 0,
        }
    
    total_corsstest_results = {
        'mse': 0,
        'rho': 0,
        'tau': 0,
        'prec_at_10': 0,
        'prec_at_20': 0,
    }
    
    
    for num in range(1,6):    
        print('fold{}:\n'.format(num))
        flag=0 #flag=1:cross_dataset
        dataset = GraphRegressionDataset(args,num,flag)
        if args.cross_test == 'yes': 
            flag=1 #flag=1:cross_dataset
            cross_dataset = GraphRegressionDataset(args,num,flag)
        args.num_features = dataset.number_features
        if args.model == 'gnn_base':
           model = gnnbasedModel(args).to(args.device)
        elif args.model == 'mlp_base':
           model = mlpbasedModel(args).to(args.device)
        elif args.model == 'SAT_GSL':
           model =SAT_GSL(args).to(args.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        train()
        model.load_state_dict(torch.load(model_save_path))
        test_results=evaluate()
        for key,value in test_results.items():
            if key in total_test_results:
                total_test_results[key] += value
            else:
                total_test_results[key] = value

        if args.cross_test == 'yes': 
            corsstest_results=evaluate_crossdata()
            for key,value in corsstest_results.items():
                if key in total_corsstest_results:
                    total_corsstest_results[key] += value
                else:
                    total_corsstest_results[key] = value

    print('\n5fold avg result:')
    for key,value in total_test_results.items():
        print('\t {} = {}'.format(key,value/5))

    if args.cross_test == 'yes': 
        print('\n5fold avg result：')
        for key,value in total_corsstest_results.items():
            print('\t {} = {}'.format(key,value/5))
