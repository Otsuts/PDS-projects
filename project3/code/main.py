import argparse
from trainer import SemEmb, SynTrainer, SemRel
from utils import set_seed, write_log


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--eval_iter', type=int, default=1)
    parser.add_argument('--model', type=str, default='semantic_relatedness')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_discriminator', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_lr', type=float, default=1e-6)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--cls_lr', type=float, default=1e-4)
    parser.add_argument('--fcls_lr', type=float, default=1e-4)
    parser.add_argument('--num_samples', default=400, type=int)
    parser.add_argument('--gan_epochs', type=int, default=40)
    parser.add_argument('--use_big', action='store_false')
    return parser.parse_args()


def main(args):
    if args.model == 'semantic_embedding':
        trainer = SemEmb(args)
    elif args.model == 'synthetic':
        trainer = SynTrainer(args)
    elif args.model == 'semantic_relatedness':
        trainer = SemRel(args)
    else:
        raise f'Model {args.model} not supported'
    trainer.train()


if __name__ == '__main__':
    args = get_args()
    # 内含随机性算法，不能seetseed()
    if args.model != 'synthetic':
        set_seed()
    main(args)
