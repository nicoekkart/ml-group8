
# TODO: Create a class Logger that also writes model parameters at the beginning
def write_loss(writer, epoch, loss, train):
    label = 'Train' if train else 'Val'
    writer.add_scalar(label + '/Loss', loss, epoch)
    print('Epoch {} - {} Loss: {:.4f}'.format(epoch, label, loss))