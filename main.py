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
    person_df.to_csv(path[:-4] + '_person.csv')
    item_df.to_csv(path[:-4] + '_item.csv')

def main(path):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    params = {'batch_size': 512, 
              'shuffle': True,
              'num_workers': 16,
              'pin_memory': True}

    max_epochs = 100

    data = util.IRTPDataset(path)
    loader = torch.utils.data.DataLoader(data, **params)

    model = IRTP(data.n_persons, data.n_items, 2)
    model.to(device)

    loss_fn = nn.BCELoss()
    
    e_lr = 0.1
    m_lr = 0.01
    e_params = []
    m_params = []

    for idx, p in enumerate(model.parameters()):
        if idx < 3:
            e_params += [{'params':p, 'lr':e_lr}]
        else:
            m_params += [{'params':p, 'lr':m_lr}]

    e_opt = torch.optim.SGD(e_params)
    m_opt = torch.optim.SGD(m_params)

    for epoch in range(max_epochs):
        print('Epoch: {0}'.format(epoch))

        i, loss = em_step(e_opt, loader, model, loss_fn, device)
        print('  E-Step | Iter: {0} | Loss: {1}'.format(i, loss))

        i, loss = em_step(m_opt, loader, model, loss_fn, device)
        print('  M-Step | Iter: {0} | Loss: {1}'.format(i, loss))


    write_output(model, path)

if __name__ == '__main__':
    main('data/tiny_test.csv')

