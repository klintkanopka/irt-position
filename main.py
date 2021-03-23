import argparse
import torch
import torch.nn as nn
import numpy as np
from model import IRTP
import util
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def em_step(optimizer, loss, loader, model, loss_fn, device, max_iter=100, eps=1e-2, verbose=False):
    loss_log = []
    for i in tqdm(range(max_iter), desc='Iteration'):
        last_loss = loss
        loss = n_samples = 0
        for batch, resp in loader:
            batch, resp = batch.to(device), resp.to(device)

            optimizer.zero_grad()
            y = model(batch)
            batch_loss = loss_fn(y, resp.double())
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item() * batch.shape[0]
            n_samples += batch.shape[0]

        loss = loss / n_samples
        loss_log += [loss]

        if i>0 and np.abs(last_loss - loss) < eps:
            return i, loss, loss_log
        elif verbose:
            print('    iter: {0} | loss: {1}'.format(i, loss))

    return i, loss, loss_log

def write_output(model, loss_log, path_stem):

    person_dict = {}
    item_dict = {}
    loss_dict = {'loss': loss_log}

    for idx, p in enumerate(model.named_parameters()):
        if idx < 3:
            person_dict[p[0]] = p[1].data.cpu().numpy()
        else:
            item_dict[p[0]] = p[1].data.cpu().numpy()

    person_df = pd.DataFrame(person_dict)
    person_df.index.name = 'sid'
    item_df = pd.DataFrame(item_dict)
    item_df.index.name = 'ik'
    loss_df = pd.DataFrame(loss_dict)
    loss_df.index.name = 'epoch'

    person_key = pd.read_csv(path_stem + '_person_key.csv')
    item_key = pd.read_csv(path_stem + '_item_key.csv')

    person_df = pd.merge(person_key, person_df, on='sid', how='inner')
    item_df = pd.merge(item_key, item_df, on='ik', how='inner')

    person_df.to_csv(path_stem + '_person_params.csv', index=False)
    item_df.to_csv(path_stem + '_item_params.csv', index=False)
    loss_df.to_csv(path_stem + '_loss.csv')

def main(run_name, input_path, irt_model, max_epochs, max_iter, 
            batch_size, num_workers, learning_rate_E, 
            learning_rate_M, epsilon, verbose):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    h_params ={'input_path': input_path,
               'irt_model': irt_model,
               'max_epochs': max_epochs,
               'max_iter': max_iter,
               'batch_size': batch_size,
               'lr_E': learning_rate_E,
               'lr_M': learning_rate_M,
               'epsilon': epsilon}


    params = {'batch_size': batch_size, 
              'shuffle': True,
              'num_workers': num_workers,
              'pin_memory': True}

    data = util.IRTPDataset(input_path, run_name)
    loader = torch.utils.data.DataLoader(data, **params)

    tb = SummaryWriter()

    model = IRTP(data.n_persons, data.n_items, irt_model)
    tmp_data, _ = iter(loader).next()
    tb.add_graph(model, tmp_data)
    #tb.add_hparams(h_params)
    tb.flush()
    model.to(device)

    loss_fn = nn.BCELoss()
    
    e_params = []
    m_params = []

    for idx, p in enumerate(model.parameters()):
        if idx < 3:
            e_params += [{'params':p, 'lr':learning_rate_E}]
        else:
            m_params += [{'params':p, 'lr':learning_rate_M}]

    e_opt = torch.optim.SGD(e_params)
    m_opt = torch.optim.SGD(m_params)

    epoch = 0
    last_loss = 1e6 
    loss = 1e5
    loss_log = []

    while epoch <= max_epochs and np.abs(last_loss - loss) > epsilon:
        
        print('Epoch: {0}'.format(epoch))
        last_loss = loss

        i, loss, log = em_step(e_opt, loss, loader, model, loss_fn, device, max_iter, epsilon, verbose)
        print('  E-Step | Iter: {0} | Loss: {1}'.format(i, loss))

        tb.add_scalar('Loss', loss, 2*epoch)
        tb.add_histogram('theta', model.theta, 2*epoch)
        tb.add_histogram('k', model.k, 2*epoch)
        tb.add_histogram('c', model.c, 2*epoch)
        tb.flush()

        loss_log += log

        i, loss, log = em_step(m_opt, loss, loader, model, loss_fn, device, max_iter, epsilon, verbose)
        print('  M-Step | Iter: {0} | Loss: {1}'.format(i, loss))

        tb.add_scalar('Loss', loss, 2*epoch + 1)
        tb.add_histogram('beta_early', model.beta_e, 2*epoch+1)
        tb.add_histogram('beta_late', model.beta_l, 2*epoch+1)
        tb.add_histogram('alpha_early', model.alpha_e, 2*epoch+1)
        tb.add_histogram('alpha_late', model.alpha_l, 2*epoch+1)
        tb.flush()

        loss_log += log
        epoch += 1

    write_output(model, loss_log, data.path_stem)
    tb.close()

if __name__ == '__main__':
    desc = 'Software for fitting an IRT mixture model for position effects'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-N', '--name', help='run name', type=str, default=None)
    parser.add_argument('-i', '--input', help='input file path', default='data/tiny_test.csv')
    parser.add_argument('-m', '--irt_model', help='number of parameters for IRT model - pick 1 or 2', type=int, default=2)
    parser.add_argument('-e', '--epochs', help='max epochs', type=int, default=100)
    parser.add_argument('-n', '--max_iterations', help='max iterations within each E/M step', type=int, default=100)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('-w', '--num_workers', help='number of parallel workers for data loader', type=int, default=2)
    parser.add_argument('-E', '--learning_rate_E', help='E step learning rate', type=float, default=0.1)
    parser.add_argument('-M', '--learning_rate_M', help='M step learning rate', type=float, default=0.01)
    parser.add_argument('-c', '--epsilon', help='epsilon for convergence', type=float, default=0.001)
    parser.add_argument('-v', '--verbose', help='print additional fitting information', action='store_true', default=False)
    args = parser.parse_args()

    if args.name is not None:
        print('\nFitting {}: {}PL Model to {} with convergence threshold of {}\n'.format(args.name, args.irt_model, args.input, args.epsilon))
    else:
        print('\nFitting {}PL Model to {} with convergence threshold of {}\n'.format(args.irt_model, args.input, args.epsilon))
    
    main(args.name, args.input, args.irt_model, args.epochs, args.max_iterations, 
            args.batch_size, args.num_workers, args.learning_rate_E, 
            args.learning_rate_M, args.epsilon, args.verbose)

