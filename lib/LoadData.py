import pandas as pd
from mylogging import mylogger


def load_train_data():
    logger = mylogger()
    logger.info('read_train start')
    train = pd.read_csv('../input/train.csv')
    logger.info('train:\n{}'.format(train.head()))
    logger.debug('read_train end')
    return train


def load_test_data():
    logger = mylogger()
    logger.info('read_test start')
    test = pd.read_csv('../input/test.csv')
    logger.info('test:\n{}'.format(test.head()))
    logger.debug('read_test end')
    return test


def load_submission():
    logger = mylogger()
    logger.info('read_submission start')
    submission = pd.read_csv('../input/sample_submission.csv')
    logger.info('sample_submission:\n{}'.format(submission.head()))
    logger.debug('read_submission end')
    return submission


def load_suppliment():
    logger = mylogger()
    logger.info('read_csv start')
    item_cat = pd.read_csv('../input/item_categories.csv')
    item = pd.read_csv('../input/items.csv')
    shop = pd.read_csv('../input/shops.csv')
    logger.debug('read_csv end')
    logger.info('item_cat:\n{}'.format(item_cat.head()))
    logger.info('item:\n{}'.format(item.head()))
    logger.info('shop:\n{}'.format(shop.head()))
    return item_cat, item, shop


if __name__ == '__main__':
    load_train_data()
    load_test_data()
    #load_submission()
