import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DCLKR")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="music", help="Choose a dataset:[music,movie,book]")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")
    parser.add_argument("--model_path", nargs="?", default="model/best_fm.ckpt", help="path for pretrain model")

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument("--n_factors", type=int, default=2, help="number of disentangled aspects")
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument('--lambda1', type=float, default=1e-2, help='intra-view contrastive loss weight')
    parser.add_argument('--lambda2', type=float, default=1e-2, help='inter-view contrastive loss weight')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[5, 10, 20, 50, 100]', help='Output sizes of every layer')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No, pretrain, 1: Pretrain with the learned embeddings, 2:Pretrain with stored models.')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    return parser.parse_args()
