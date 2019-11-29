import argparse
import datahelper
import model
from header import *




def defineArgs():
    """define args"""
    parser = argparse.ArgumentParser(description = "Chinese_poem_generator.")
    parser.add_argument("-m", "--mode", help = "select mode by 'train' or test or head",
                        choices = ["train", "test", "head"], default = "train")
    return parser.parse_args()

if __name__ == "__main__":
    X, Y, wordNum, wordToID, words = datahelper.readpoems(trainPoems)
    args = defineArgs()
    if args.mode == "train":
        print("1 training...")
        model.train(X, Y, wordNum)
    else:
        if args.mode == "test":
            print("2 genrating...")
            poems = model.test(wordNum, wordToID, words)
        else:
            characters = input("please input chinese character:")
            print("3 genrating...")
            poems = model.testHead(wordNum, wordToID, words, characters)