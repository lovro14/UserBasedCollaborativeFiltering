from utils.config import parse_args
from utils.data_loader import DataLoader
from recommender.recommender_system import RecommenderSystem
from evaluator.recommender_evaluator import RecommenderEvaluator
import numpy as np


def main(args):
    dataset = args.dataset
    neighborhood_size = args.neighborhood_size
    recommended_list_size = args.recommended_list_size

    data_loader = DataLoader(dataset)
    data_loader.load_data()
    user_number, item_number = data_loader.get_dataset_info()
    train, test = data_loader.train_test_split()
    recommender = RecommenderSystem()
    rating_predictions = recommender.predict_topk_nobias(train, k=neighborhood_size)

    evaluator = RecommenderEvaluator()
    print("RMSE={}".format(evaluator.rmse(rating_predictions, test)))
    print("MAE={}".format(evaluator.mae(rating_predictions, test)))
    mean_test = np.true_divide(test.sum(1), (test != 0).sum(1))
    precisions, recalls = evaluator.precision_recall_at_k(rating_predictions, test,
                                                          mean_test, user_number, recommended_list_size)
    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    f1 = evaluator.f1(precision, recall)
    print("Precision({})={}".format(recommended_list_size, precision))
    print("Recall({})={}".format(recommended_list_size, recall))
    print("F1({})={}".format(recommended_list_size, f1))


if __name__ == '__main__':
    args = parse_args()
    main(args)
