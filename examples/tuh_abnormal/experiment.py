from brainfeaturedecode.data_set.tuh_abnormal import TuhAbnormalTrain, \
    TuhAbnormalEval
from brainfeaturedecode.experiment.experiment import Experiment


def main():
    train_set = TuhAbnormalTrain(
        "/home/gemeinl/data/tuh-abnormal-eeg/raw/v2.0.0/", extension=".edf")
    exp = Experiment(train_set)
    exp.run()

    eval_set = TuhAbnormalEval(
        "/home/gemeinl/data/tuh-abnormal-eeg/raw/v2.0.0/", extension=".edf")
    exp2 = Experiment(train_set, eval_set=eval_set)
    exp2.run()

    return exp, exp2


if __name__ == '__main__':
    main()
