# TODO: Create a class Logger that also writes out allmodel parameters at the beginning

def write_loss(writer, fold, epoch, loss, train):
    label = 'Train' if train else 'Val'
    # writer.add_scalar(label + '/Loss', loss, epoch)
    print('Fold {} - Epoch {} - {} Loss: {:.4f}'.format(fold, epoch, label, loss))

