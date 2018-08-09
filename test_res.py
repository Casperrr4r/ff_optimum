# -*- coding: utf-8 -*-
from ff_optimum.cores.optimizer import OptimizerFactory


def main():

    charge_path = (
        '/home/casper/ff_optimum/configs/single_sa/config_all.json')

    factory = OptimizerFactory()

    optimizer = factory.create_optimizer_from_config(charge_path)

    # optimizer.train()

    optimizer.test()

    # optimizer.save_parameters_to_file()


if __name__ == '__main__':
    main()
