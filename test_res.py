# -*- coding: utf-8 -*-
from ff_optimum.cores.optimizer import OptimizerFactory


def main():

    factory = OptimizerFactory()

    optimizer = factory.create_optimizer_from_config(sys.argv[1])

    optimizer.test()


if __name__ == '__main__':
    main()
