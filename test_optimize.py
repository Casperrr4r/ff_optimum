# -*- coding: utf-8 -*-
import sys

from ff_optimum.cores.optimizer import OptimizerFactory


def main():

    optimizer = OptimizerFactory().create_optimizer_from_config(sys.argv[1])

    optimizer.optimize()

    # optimizer.test()

    optimizer.save_parameters_to_file('ffield')


if __name__ == '__main__':
    main()
