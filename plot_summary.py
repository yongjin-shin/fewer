from plot_figure import *
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))


class SubFolderOrganizer:
    def __init__(self, root):
        self.root = root
        self.dirs = np.sort(next(os.walk(root))[1])

    def get_exp(self):
        ret = []
        for _dir in self.dirs:
            sub_root = f"{self.root}{_dir}/"
            organizer = FolderOrganizer(sub_root)
            organizer.parser()
            ret.append({
                'root': sub_root,
                'folders': organizer.get_exp_dirs(),
                'order': _dir
            })
        return ret


def get_results():
    pass


def main(arg):
    test = SubFolderOrganizer(arg.root)
    exp_dirs = test.get_exp()

    x_cost = {'vanilla': [], 'pruning': [], 'FEWER': []}
    y_acc = {'vanilla': [], 'pruning': [], 'FEWER': []}
    _order = []
    for exp in exp_dirs:
        _order.append(exp['order'])
        data = read_all(exp['root'], exp['folders'], arg)
        for d in data:
            p = d['path'].split('/')[-1]
            if 'vanilla' in p:
                x_cost['vanilla'].append(d['cost']['mean'][-1])
                y_acc['vanilla'].append(d['acc']['mean'][-1])
            elif 'reverse' in p:
                x_cost['FEWER'].append(d['cost']['mean'][-1])
                y_acc['FEWER'].append(d['acc']['mean'][-1])
            else:
                x_cost['pruning'].append(d['cost']['mean'][-1])
                y_acc['pruning'].append(d['acc']['mean'][-1])

    for i in range(len(arg.legend)):
        if 'cost' in arg.root:
            x, y, _os = del_offset(
                arg, np.copy(x_cost['vanilla']), np.copy(y_acc[arg.legend[i]]), np.copy(_order)
            )
        else:
            x, y, _os = del_offset(
                arg, np.copy(x_cost[arg.legend[i]]), np.copy(y_acc[arg.legend[i]]), np.copy(_order)
            )
        plt.plot(x, y, marker='o', markersize=3, label=legend_changer(arg, i))

    plt.legend(loc='lower left')
    plt.xlabel('Cost(Mb)', fontweight='bold', fontsize=12)
    plt.ylabel('Acc(%)', fontweight='bold', fontsize=12)
    plt.ylim(args.ylim)
    plt.tight_layout()
    if 'mnist' in exp_dirs[0]['root']:
        plt.savefig(f"{args.root}mnist_summary.png", dpi=300)
    else:
        plt.savefig(f"{args.root}cifar_summary.png", dpi=300)
    plt.show()
    plt.close()

    for i in range(len(arg.legend)):
        if 'cost' in arg.root:
            x, y, _os = del_offset(
                arg, np.copy(x_cost['vanilla']), np.copy(y_acc[arg.legend[i]]), np.copy(_order)
            )
        else:
            x, y, _os = del_offset(
                arg, np.copy(x_cost[arg.legend[i]]), np.copy(y_acc[arg.legend[i]]), np.copy(_order)
            )

        plt.plot(x, y, marker='o', markersize=3, label=legend_changer(arg, i))
        for (o, x, y) in zip(_os, x, y):
            plt.annotate(o, (x, y), textcoords="offset points", xytext=(0, 2), ha='center')
    plt.legend(loc='lower left')
    plt.xlabel('Cost(Mb)', fontweight='bold', fontsize=12)
    plt.ylabel('Acc(%)', fontweight='bold', fontsize=12)
    plt.ylim(args.ylim)
    plt.tight_layout()
    if 'mnist' in exp_dirs[0]['root']:
        plt.savefig(f"{args.root}mnist_summary_with_annotation.png")
    else:
        plt.savefig(f"{args.root}cifar_summary_with_annotation.png")
    plt.close()


def del_offset(arg, xs, ys, _os):
    drop_idx = np.where(ys < arg.ylim[0])[0]
    xs = np.delete(xs, drop_idx)
    ys = np.delete(ys, drop_idx)
    _os = np.delete(_os, drop_idx)
    return xs, ys, _os


def legend_changer(arg, i):
    if arg.legend[i] == 'FEWER':
        return f"{arg.legend[i]}(Ours)"
    elif arg.legend[i] == 'pruning':
        return f"Pruning"
    else:
        return 'FedAvg'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot parser')
    parser.add_argument('--legend', default=['vanilla', 'pruning', 'FEWER'], nargs='+')
    parser.add_argument('--title', type=str)
    parser.add_argument('--xs', default=['round'], nargs='+')
    parser.add_argument('--ys', default=['cost', 'acc'], nargs='+', type=str)
    parser.add_argument('--root', default='./log/FEWER/main_exp/cifar/', type=str)
    parser.add_argument('--ylim', default=[55, 72], nargs='+', type=str)
    # parser.add_argument('--root', default='./log/FEWER/same_sparsity/mnist/', type=str)
    # parser.add_argument('--ylim', default=[95, 100], nargs='+', type=str)
    args = parser.parse_args()

    main(args)
