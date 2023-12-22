# coding=utf-8
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--dataset_root',
                        type=str,
                        help='path to dataset',
                        default='.' + os.sep + 'dataset')

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='.' + os.sep + 'output')

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=20)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=5)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-its_Tr', '--iterations_train',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    parser.add_argument('-its_Val', '--iterations_val',
                        type=int,
                        help='number of episodes per epoch, default=10',
                        default=10)

    parser.add_argument('-pred_cTr', '--pred_classes_per_it',
                        type=int,
                        help='number of random classes for predicate sampling based training, default=49 (max_val=49)',
                        default=49)

    parser.add_argument('-obj_cTr', '--obj_classes_per_it',
                        type=int,
                        help='number of random classes object sampling based training, default=150  (max_val=150)',
                        default=150)

    parser.add_argument('-pred_nsTr', '--pred_num_samples',
                        type=int,
                        help='number of samples per class to use predicate sampling based training, default=5',
                        default=5)

    parser.add_argument('-obj_nsTr', '--obj_num_samples',
                        type=int,
                        help='number of samples per class to use predicate sampling based training, default=2',
                        default=2)

    parser.add_argument('-batch_size', '--batch_size',
                        type=int,
                        help='number of samples within each batch, default=256',
                        default=256)

    parser.add_argument('-init_weight_path', '--init_weight_path',
                        type=str,
                        help='path to the initial model weights, default=None',
                        default=None)

    parser.add_argument('-train_task', '--train_task',
                        type=int,
                        help='input of the task for the dt2 training, [possible options: {1,2}], default=1 \n- Task-1: PredCls\n- Task-2: SGCls',
                        default=1)

    parser.add_argument('-stage', '--stage',
                        type=int,
                        help='input of the stage for the dt2 training, [possible options: {1,2}], default=1 \n- Stage-1: Standard Random Sampling\n- Stage-2: Class Balanced Sampling',
                        default=1)

    parser.add_argument('-type', '--type',
                        type=int,
                        help='input of the type for the ACBS stage-2 training, [possible options: {0,1}], default=0 \n- Type-0: Use Object ground truth labels for predicate prediction\n- Type-1: Use Object classifier labels for predicate prediction',
                        default=0)

    parser.add_argument('-alpha', '--alpha',
                        type=float,
                        help='input for scaling the knowledge distillation loss in E-step, default=0.2',
                        default=0.2)

    parser.add_argument('-beta', '--beta',
                        type=float,
                        help='input for scaling the teahcer head loss in P-step, default=1.0',
                        default=1.0)

    parser.add_argument('-temperature', '--temperature',
                        type=int,
                        help='input for the temprature in sofmax for knowledge distillation, default=10',
                        default=10)

    parser.add_argument('--non_soft_label_kd',
                        action='store_true',
                        help='disables soft label based knowledge distillation and uses weight-norm based regularization for ACBS')

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=5)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')

    return parser

def get_parser_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--dataset_root',
                        type=str,
                        help='path to dataset',
                        default='.' + os.sep + 'dataset')

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='.' + os.sep + 'output')

    parser.add_argument('-init_weight_path', '--init_weight_path',
                        type=str,
                        help='path to the initial model weights, default=None',
                        default=None)

    parser.add_argument('-test_task', '--test_task',
                        type=int,
                        help='input of the task for the dt2 evaluation, [possible options: {1,2,3}], default=1 \n- Task-1: PredCls\n- Task-2: SGCls\n- Task-3: SGDet',
                        default=1)

    parser.add_argument('-batch_size', '--batch_size',
                        type=int,
                        help='number of samples within each batch, default=256',
                        default=256)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=5)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')

    return parser
