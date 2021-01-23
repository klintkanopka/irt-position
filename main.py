import argparse
import torch
import torch.nn as nn
from model import IRTP
import util
import pandas as pd


def em_step(optimizer, loader, model, loss_fn, device, max_iter=100, eps=1e-2, verbose=False):
    loss = 0
    for i in range(max_iter):
        last_loss = loss
        loss = 0
        for batch, resp in loader:
            batch, resp = batch.to(device), resp.to(device)

            optimizer.zero_grad()
            y = model(batch)
            batch_loss = loss_fn(y, resp.float())
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item() * batch.shape[0]

        if i > 0 and last_loss - loss < eps:
            return i, loss
        elif verbose:
            print('    iter: {0} | loss: {1}'.format(i, loss))

    return i, loss

def write_output(model, path):
    person_dict = {}
    item_dict = {}
    for idx, p in enumerate(model.named_parameters()):
        if idx < 3:
            person_dict[p[0]] = p[1].data.cpu().numpy()
        else:
            item_dict[p[0]] = p[1].data.cpu().numpy()
    print(person_dict, item_dict)
    person_df = pd.DataFrame(person_dict)
    item_df = pd.DataFrame(item_dict)
    person_df.to_csv(path[:-4] + '_person_params.csv')
    item_df.to_csv(path[:-4] + '_item_params.csv')

def main(input_path, irt_model, max_epochs, max_iter, 
            batch_size, num_workers, learning_rate_E, 
            learning_rate_M, epsilon, verbose):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    params = {'batch_size': batch_size, 
              'shuffle': True,
              'num_workers': num_workers,
              'pin_memory': True}

    data = util.IRTPDataset(input_path)
    loader = torch.utils.data.DataLoader(data, **params)

    model = IRTP(data.n_persons, data.n_items, irt_model)
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

    for epoch in range(max_epochs):
        print('Epoch: {0}'.format(epoch))

        i, loss = em_step(e_opt, loader, model, loss_fn, device, max_iter, epsilon, verbose)
        print('  E-Step | Iter: {0} | Loss: {1}'.format(i, loss))

        i, loss = em_step(m_opt, loader, model, loss_fn, device, max_iter, epsilon, verbose)
        print('  M-Step | Iter: {0} | Loss: {1}'.format(i, loss))


    write_output(model, input_path)

if __name__ == '__main__':
    desc = 'Software for fitting an IRT mixture model for position effects'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', help='input file path', default='data/tiny_test.csv')
    parser.add_argument('-m', '--irt_model', help='number of parameters for IRT model - pick 1 or 2', default=2)
    parser.add_argument('-e', '--epochs', help='max epochs', default=100)
    parser.add_argument('-n', '--max_iterations', help='max iterations within each E/M step', default=100)
    parser.add_argument('-b', '--batch_size', help='batch size', default=128)
    parser.add_argument('-w', '--num_workers', help='number of parallel workers for data loader', default=2)
    parser.add_argument('-E', '--learning_rate_E', help='E step learning rate', default=0.1)
    parser.add_argument('-M', '--learning_rate_M', help='M step learning rate', default=0.01)
    parser.add_argument('-c', '--epsilon', help='epsilon for convergence', default=0.01)
    parser.add_argument('-v', '--verbose', help='print additional fitting information', action='store_true', default=False)
    args = parser.parse_args()
    main(args.input, args.irt_model, args.epochs, args.max_iterations, 
            args.batch_size, args.num_workers, args.learning_rate_E, 
            args.learning_rate_M, args.epsilon, args.verbose)

