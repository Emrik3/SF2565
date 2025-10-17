import torch
data = torch.load('data/train.pt')
labels = data['labels']
print('Label shape:', labels.shape)
print('\nFirst 30 samples:')
print(labels[:30])
print('\nColumn 0 unique:', torch.unique(labels[:, 0]))
print('Column 1 unique:', torch.unique(labels[:, 1]))
print('\nAre columns identical?', torch.equal(labels[:, 0], labels[:, 1]))
