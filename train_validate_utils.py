import torch
import sys


def validate(**config):
    correct = 0
    total = 0
    config['net'].eval()
    val_loss = 0
    with torch.no_grad():
        for data in config['val_loader']:
            images, labels = data
            outputs = config['net'](images)

            # validation loss calculation
            loss = config['criterion'](outputs, labels)
            val_loss +=loss.item()

            # validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    return val_loss/total, val_acc


def train(**config):
    min_loss = float('inf')
    for epoch in range(config['epochs']):
        train_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(config['train_loader'], 0):
            inputs, labels = data

            config['optimizer'].zero_grad()

            outputs = config['net'](inputs)
            loss = config['criterion'](outputs, labels)
            loss.backward()
            config['optimizer'].step()
            train_loss += loss.item()

            # train accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print progress
            sys.stdout.write("\rTrain:[Batch %d/%d]" % (i, len(config['train_loader'])))
        train_acc = correct/total
        train_loss = train_loss/total
        val_loss, val_acc = validate(**config)
        print('\n[Epoch %d] ---- train_loss: %.5f ---- val_loss: %.5f ---- train_acc: %.5f ---- val_acc: %.5f'
               % (epoch + 1, train_loss,val_loss,train_acc,val_acc))
        # if min_loss > val_loss:
        #     torch.save(config['net'].state_dict(), config['model_path'])
        #     min_loss = val_loss
        #logging.info('Epoch {} ---- train_loss {} ---- val_loss {} ---- train acc {} ----- val acc {}'
        #         .format(epoch, train_loss, val_loss, train_acc, val_acc))
