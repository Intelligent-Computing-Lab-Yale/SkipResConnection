import config
from models.model_snndirect import *
# from model import *


from utils import *

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
device = torch.device("cuda")
cudnn.benchmark = True
cudnn.deterministic = True



def main(args):

    print (args.t_init)
    args = config.get_args()

    train_loader, test_loader = data_load(args)

    criterion = torch.nn.CrossEntropyLoss()
    weight_sum_K = 1
    scale = args.t_scale

    print ("t scale:", scale)

    if args.arch == 'base':
        model = mid_vgg_direct(max_t=scale)
    elif args.arch == 'res':
        model = mid_vgg_direct_residual(max_t=scale)
    elif args.arch == 'shuffle':
        model = mid_shufflenet_direct(max_t=scale, t_init = args.t_init)

    model = model.cuda()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,eta_min=0)



    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [losses])


    all_testacclist = []
    all_trainacclist = []
    all_losslist = []
    all_delaymeanlist = []
    all_delaystdlist = []

    delay_mean = torch.mean(torch.stack([model.t_shift1.flatten(), model.t_shift2.flatten()]))
    delay_std = torch.std(torch.stack([model.t_shift1.flatten(), model.t_shift2.flatten()]))
    print ('init delay..')
    print (delay_mean, delay_std)
    for epoch in range(args.epochs):
        model.train()

        epoch_loss = []
        for i, (data, target) in enumerate(train_loader):
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()

            if args.arch == 'shuffle':
                outs,weight_sum_cost, force_loss = model(data)
            else:
                outs,weight_sum_cost  = model(data)
                force_loss = 0

            ce_loss = criterion(-1*(outs), target)

            reg_loss = weight_sum_K*weight_sum_cost
            loss = ce_loss+ reg_loss + force_loss*1e-6
            losses.update((torch.mean(ce_loss)).item(), data.shape[0])

            loss.sum().backward()
            epoch_loss.append(float(ce_loss.data.cpu().numpy()))

            optimizer.step()

            if (i+1) % 1500 == 0:    # print every 2000 mini-batches
                progress.display(i)
        scheduler.step()

        all_losslist.append(sum(epoch_loss)/len(epoch_loss))
        if (epoch+1) % 5 ==0:
            acc = test(model, test_loader, epoch,args)
            acc_train = test(model, train_loader, epoch,args)

            all_testacclist.append(acc)
            all_trainacclist.append(acc_train)

            if args.arch == 'shuffle':
                delay_mean = torch.mean(torch.stack([model.t_shift1.flatten(), model.t_shift2.flatten()]))
                delay_std = torch.std(torch.stack([model.t_shift1.flatten(), model.t_shift2.flatten()]))
                all_delaymeanlist.append(float(delay_mean.cpu().data.numpy()))
                all_delaystdlist.append(float(delay_std.cpu().data.numpy()))

    print (all_losslist)
    print(all_testacclist)

    torch.save(model.state_dict(), 'savemodel/{}_{}_init{}_t{}_b{}_lr{}_ep{}'.format(args.dataset, args.arch, args.t_init, args.t_scale, args.batch_size, args.lr, args.epochs))

    return np.max(all_testacclist)

def test(model, test_loader, epoch,args):

    model.eval()
    correct = 0
    total = 0
    first_spike_time_list = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # the class with the highest energy is what we choose as prediction
            first_spike_time, predicted = torch.min(outputs.data, 1)
            first_spike_time_list.append(torch.log(first_spike_time).sum() / first_spike_time.size(0))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print("first_spike mean time:", sum(first_spike_time_list) / len(first_spike_time_list))


    return (100 * correct / total)

if __name__ == '__main__':
    args = config.get_args()
    main(args)

