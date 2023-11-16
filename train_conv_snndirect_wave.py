import config
import pickle

from models.model_wave import *
from torchvision.transforms import ToTensor, ToPILImage, Resize
from utils import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda")


torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
cudnn.benchmark = True
cudnn.deterministic = True

class CustomDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx].astype(np.float32)
        label = self.labels[idx]

        if self.transform:
            input_data = self.transform(input_data)

        return input_data, label


def main(args):

    print (args.t_init)
    args = config.get_args()

    label_num = 36
    # Load the dataset from the saved file
    with open('wave/dataset_wave_{}.pkl'.format(label_num), 'rb') as f:
        loaded_dataset = pickle.load(f)
    loaded_inputs = loaded_dataset['inputs']
    loaded_labels = loaded_dataset['labels']

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        loaded_inputs, loaded_labels, test_size=0.2, random_state=1234
    )


    transform_wave = transforms.Compose([
        ToPILImage(),       # Convert to PIL Image
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    trainset = CustomDataset(inputs=train_inputs, labels=train_labels, transform=transform_wave)
    valset = CustomDataset(inputs=test_inputs, labels=test_labels, transform=transform_wave)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, pin_memory=False, num_workers=4)

    test_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                              shuffle=False, pin_memory=False, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()
    weight_sum_K = 1
    scale = args.t_scale

    print ("t scale:", scale)

    if args.arch == 'base':
        model = mid_vgg_direct_wave(max_t=scale, n_class=label_num)
    elif args.arch == 'res':
        model = mid_vgg_direct_residual_wave(max_t=scale, n_class=label_num)
    elif args.arch == 'shuffle':
        model = mid_shufflenet_direct_wave(max_t=scale, t_init = args.t_init, n_class=label_num)

    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,eta_min=0)


    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [losses])


    all_acclist = []
    all_losslist = []
    for epoch in range(args.epochs):
        model.train()


        epoch_loss = []
        for i, (data, target) in enumerate(train_loader):
            data = data.float().cuda()
            target = target.cuda()
            optimizer.zero_grad()


            if args.arch == 'shuffle':
                outs,weight_sum_cost, force_loss = model(data)
            else:
                outs, weight_sum_cost = model(data)
            ce_loss = criterion(-1*(outs), target)

            reg_loss = weight_sum_K*weight_sum_cost
            loss = ce_loss+ reg_loss#+ force_loss*1e-6
            losses.update((torch.mean(ce_loss)).item(), data.shape[0])

            loss.sum().backward()
            epoch_loss.append(float(ce_loss.data.cpu().numpy()))
            optimizer.step()


            if (i+1) % 1500 == 0:
                progress.display(i)

        scheduler.step()

        all_losslist.append(sum(epoch_loss)/len(epoch_loss))
        if (epoch+1) % 10 ==0:
            acc = test(model, test_loader, epoch,args)
            all_acclist.append(acc)
    print (all_losslist, all_acclist)


    torch.save(model.state_dict(), 'savemodel/wave{}_{}_init{}_t{}_b{}_lr{}_ep{}'.format(label_num, args.arch, args.t_init, args.t_scale, args.batch_size, args.lr, args.epochs))

    return np.max(all_acclist)

def test(model, test_loader, epoch,args):
    if args.arch == 'shuffle':
        print ("tshift", torch.mean(model.t_shift1), torch.mean(model.t_shift2))
    model.eval()
    correct = 0
    total = 0
    first_spike_time_list = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.float().to(device)
            labels = labels.to(device)

            outputs = model(images)


            # the class with the highest energy is what we choose as prediction
            first_spike_time, predicted = torch.min(outputs.data, 1)
            first_spike_time_list.append(torch.log(first_spike_time).sum() / first_spike_time.size(0))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print("first_spike mean time:", sum(first_spike_time_list) / len(first_spike_time_list))
    print('Epoch:', epoch)
    print('Accuracy of the network on the 10 test images: %.3f %%' % (
            100 * correct / total))


    return (100 * correct / total)

if __name__ == '__main__':
    args = config.get_args()


    run_list = []
    for i in range(1):
        acc = main(args)
        run_list.append(acc)
    print('5runs mean {}, std {}'.format(np.mean(run_list), np.std(run_list)))


