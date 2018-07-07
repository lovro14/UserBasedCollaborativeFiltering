import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Implementation and evaluation of"
                                                 " neural network based recommender systems")
    parser.add_argument('--dataset', type=str, default='ml-1m', choices=['ml-100k', 'ml-1m', 'ml-20m'])
    parser.add_argument('--neighborhood_size', type=int, default=30, help='Neighborhood size for rating prediction')
    parser.add_argument('--recommended_list_size', type=int, default=20, help='Recommended list size')

    return check_args(parser.parse_args())


# Checking arguments
def check_args(args):
    # --neighborhood_size
    try:
        assert args.neighborhood_size >= 1
    except:
        print('Neighborhood size must be greater or equal to 1')

    # --recommended_list_size
    try:
        assert args.recommended_list_size >= 1
    except:
        print('Recommended list size must be greater or equal to 1')

    return args
