import os
import argparse
from data_loader import pip_data
from train import train_test_interface
from predict import predict

def main():

    parser = argparse.ArgumentParser()
    # parameters defined for in pip_data
    parser.add_argument("--mode", help="pip_data, train, test or predict", default="predict", type=str)
    parser.add_argument("--data_dir",  help="Data Folder", default="./data", type=str)
    parser.add_argument("--max_df", help="tfidf term: max frequency to keep in vocab", default=0.75, type=float)
    parser.add_argument("--min_df", help="tfidf term: min counts to keep in vocab", default=2, type=int)
    parser.add_argument("--min_tfidf", help="tfidf term: min tfidf to keep in vocab", default=0.1, type=float)
    parser.add_argument("--embedding_size", default=256, help="Words embeddings dimension", type=int)
    # parameters defined for train and test model
    parser.add_argument("--max_lens", default=[98,100,34,103],
                        help="a list of max lens for merged_train_test,train_X,train_y,test_X", nargs='+', type=int)
    parser.add_argument("--batch_sz", default=128, help="batch size", type=int)
    parser.add_argument("--test_percent", default=0.05, help="proportion of test samples", type=float)
    # encoder is bidirectional gru_unit/2 for one direction
    parser.add_argument("--gru_units", default=512, help="Encode and decode GRU cell units number", type=int)
    parser.add_argument("--att_units", default=64, help="attention weights", type=int)
    parser.add_argument("--learning_rate", default=0.001, help="Learning rate", type=float)
    parser.add_argument("--clipvalue", default=2.0, help="gradient clip value", type=float)
    parser.add_argument("--checkpoint_dir", help="Checkpoint directory", default='./training_checkpoints', type=str)
    parser.add_argument("--save_chkp_epoch", help="Checkpoint save every # epoch", default=5, type=int)
    parser.add_argument("--use_checkpoint", help="for train and test, restore from checkpoint?", default=True, type=bool)
    parser.add_argument("--train_epoch", help="train epoch", default=15, type=int)
    parser.add_argument("--cov_loss_wt", help="coverage loss weight", default=0.5, type=float)
    # parameters defined for predict
    parser.add_argument("--max_len_y", default=40, help="max words of the predicted abstract", type=int)
    parser.add_argument("--min_len_y", default=5, help="min words of the predicted abstract", type=int)
    parser.add_argument("--beam_size", default=3, help="beam size for beam search", type=int)
    parser.add_argument("--prediction_path", help="Path to save prediction results", default="./prediction.txt", type=str)

    args = parser.parse_args()
    params = vars(args)
    print(params)

    assert params["mode"] in ["pip_data","train", "test", "predict"], "The mode must be pip_data, train, test or predict"
    assert os.path.exists(params["data_dir"]), "data_dir doesn't exist"

    if params["mode"] == "pip_data":
        pip_data(params)
    elif params["mode"] in ['train','test']:
        train_test_interface(params)
    elif params["mode"] == "predict":
        predict(params)

if __name__ == "__main__":
    main()